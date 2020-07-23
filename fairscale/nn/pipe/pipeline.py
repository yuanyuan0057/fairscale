# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The pipeline parallelism of Pipe."""
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast
import threading

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers, join_workers
from fairscale.nn.model_parallel import get_pipeline_parallel_group, get_pipeline_parallel_ranks

__all__: List[str] = []


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

TBirchQueue = Queue()



def rpc_send(batch: Batch):
    globals()["TBirchQueue"].put(batch)


class TbirchSend(torch.autograd.Function):
    """Copies tensors on specific streams."""

    @staticmethod
    # type: ignore
    def forward(ctx, src_rank, dst_rank, *input: Tensor,) -> Tensors:
        assert src_rank == torch.distributed.get_rank()
        name = f"Test{dst_rank}"
        print(f"sync send func {src_rank}, {dst_rank}")

        batch = [b.cpu() for b in input]

        torch.distributed.rpc.rpc_sync(name, rpc_send, args=(batch,))
        return ()

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tensors:
        print(f"Send Backward {grad}")
        return tuple(*grad)


def tbirch_send_plain(batch: Batch, src_rank: int, dst_rank: int) -> None:
    assert src_rank == torch.distributed.get_rank()
    name = f"Test{dst_rank}"
    print(f"sync send plain {src_rank}, {dst_rank}")

    batch = [b.cpu() for b in batch]

    torch.distributed.rpc.rpc_sync(name, rpc_send, args=(batch,))


class TbirchRecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst_rank: int, tensor: Tensor) -> Tensors:
        assert dst_rank == torch.distributed.get_rank()
        print(f">> sync recv func {dst_rank}")
        result = TBirchQueue.get()
        print(f"<< sync recv func {dst_rank}")
        return tuple(r.cuda().requires_grad_() for r in result)

    @staticmethod
    # type: ignore
    def backward(ctx, *grad: Tensor,) -> Tensors:
        print(f"TbirchRecvBackward tensor {grad}")
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        if this_rank == 0:
            print("wat")
            return (None, None)

        tbirch_send_plain(grad, this_rank, ranks[ranks.index(this_rank) - 1])
        return (None, None)


def tbirch_recv_plain(dst_rank: int) -> Batch:
    assert dst_rank == torch.distributed.get_rank()
    print(f">> sync recv plain {dst_rank}")
    result = TBirchQueue.get()
    print(f"<< sync recv plain {dst_rank}")
    return [r.cuda() for r in result]


# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional["Task"]]
    OutQueue = Queue[Tuple[bool, Union[Tuple["Task", Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def clock_cycles(m: int, n: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


def backward_pre_hook(*args):
    print(f"backward_pre_hook yolo {args}")


class Pipeline:
    """The pipeline parallelism for Pipe."""

    def __init__(
        self,
        partitions: List[nn.Sequential],
        devices: List[torch.device],
        copy_streams: List[List[AbstractStream]],
        skip_layout: SkipLayout,
        checkpoint_stop: int,
    ) -> None:
        self.partitions = partitions
        self.devices = devices
        self.copy_streams = copy_streams
        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop
        (self.in_queues, self.out_queues) = (None, None)  # create_workers(devices)

    def __del__(self) -> None:
        pass  # join_workers(self.in_queues, self.out_queues)

    def run(self, batches: List[Batch]) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(get_pipeline_parallel_ranks())

        if torch.distributed.get_rank(get_pipeline_parallel_group()) != n - 1:
            pass  # partitions[0].register_backward_pre_hook(backward_pre_hook)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]

        for schedule in clock_cycles(m, 2):
            print(f"doing schedule {schedule}")
            # self.fence(batches, schedule, skip_trackers)
            self.compute(batches, schedule, skip_trackers)

    def fence(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0 and j != 0:
                depend(batches[i - 1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                copy(batches[i], prev_stream, next_stream)

    def compute(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Runs tasks with synchronization to copy streams."""
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.partitions[0].training:
            checkpoint_stop = 0

        n = len(get_pipeline_parallel_ranks())
        streams = []  # [current_stream(d) for d in devices]
        exc_info: Optional[ExcInfo] = None

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        print(f"computing {schedule}")
        for i, j in schedule:
            if j != torch.distributed.get_rank(get_pipeline_parallel_group()):
                continue

            batch = batches[i]
            assert len(partitions) == 1
            partition = partitions[0]

            phony = torch.empty(0, requires_grad=True)

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                result = TbirchRecv.apply(torch.distributed.get_rank(), phony)
                if len(result) == 1:
                    batch = Batch(result[0])
                else:
                    batch = Batch(result)

            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop
            if checkpoint:
                print(f"run check")

                def function(
                    input: TensorOrTensors,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        print(f"part {(i,j)}  run {input.size()}")
                        return partition(input)

                chk = Checkpointing(function, batch)
                task = Task(None, compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:
                print(f"run reg")

                def compute(
                    batch: Batch = batch,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> Batch:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return batch.call(partition)

                task = Task(None, compute=compute, finalize=None)
                del compute

            print(f"compute task... {(i,j)}")
            batch = task.compute()

            if j != n - 1:
                ranks = get_pipeline_parallel_ranks()
                this_rank = torch.distributed.get_rank()
                TbirchSend.apply(this_rank, ranks[ranks.index(this_rank) + 1], *batch)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            # if j != n - 1:
            #     wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            # with use_device(devices[j]):
            print(f"finalize batch... {(i,j)}")
            task.finalize(batch)

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    def back_helper(self, output):
        import code, traceback, signal

        def debug(sig, frame):
            """Interrupt running process, and provide a python prompt for
            interactive debugging."""
            d = {"_frame": frame}  # Allow access to frame object.
            d.update(frame.f_globals)  # Unless shadowed by global
            d.update(frame.f_locals)

            i = code.InteractiveConsole(d)
            message = "Signal received : entering python shell.\nTraceback:\n"
            message += "".join(traceback.format_stack(frame))
            i.interact(message)

        def listen():
            signal.signal(signal.SIGUSR1, debug)  # Register handler

        def dump_grad(output, key=""):

            segments = []
            grad = output.grad_fn
            while grad is not None:
                segments.append(str(grad))
                if len(grad.next_functions) > 0:
                    grad = grad.next_functions[0][0]
                else:
                    break

            print(f"grad {key} = {' -> '.join(segments)}")

        listen()
        o = list(output)

        grads = []
        for i, batch in enumerate(o):
            rank = torch.distributed.get_rank()
            print(f"back_helper-{rank}: recv {i} of {len(o)}")
            found = tbirch_recv_plain(torch.distributed.get_rank())
            assert len(found) == 1
            grads.append(found[0])

        print(f"back_helper-{rank} grads:\n {grads}, batch: \n{batch.tensor_or_tensors}")
        dump_grad(batch.tensor_or_tensors, "back_helper")



        try:
            torch.autograd.backward(tuple(x.tensor_or_tensors for x in o), grad_tensors=tuple(grads), retain_graph=True)
        except Exception as e:
            import traceback
            print(f"back_helper-{rank} got exception {e}, {traceback.format_exc()}")

        if False: # Enable to trigger crash
            import os
            import time
            for i in range(10):
                print(f"sleeping {i}/10 for gdb in {os.getpid()}")
                time.sleep(1)
            torch.autograd.backward(tuple(x.tensor_or_tensors for x in o), grad_tensors=tuple(grads))

        print(f"back_helper-{rank} finished {i} of {len(o)}")
