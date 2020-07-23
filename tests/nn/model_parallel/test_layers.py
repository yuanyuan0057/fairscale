# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel import layers, gather_from_model_parallel_region
from fairscale.nn.pipe import Pipe
from tests.nn.model_parallel.commons import dist_init, get_world_sizes, set_random_seed, spawn_for_all_world_sizes

from torch.distributed import rpc


def run_test_parallel_embedding(rank, model_parallel_size):
    dist_init(rank, model_parallel_size)

    if torch.distributed.get_rank() == 0:
        print("> testing parallel embedding with model parallel size {} ...".format(model_parallel_size))

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_parallel = layers.ParallelEmbedding(vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_parallel(input_data)
    loss_parallel = torch.mul(output, loss_weight).sum()
    loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_parallel.sub(loss_original).abs()
    print("   error in loss (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print("   error in loss (vocab parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad, hidden_size // model_parallel_size, 1)[
        mpu.get_model_parallel_rank()
    ]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad, vocab_size // model_parallel_size, 0)[
        mpu.get_model_parallel_rank()
    ]
    error = embedding_vocab_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print("   error in grad (vocab parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-12, "error: {}".format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(">> passed the test :-)")


def run_test_initialize_affine_weight(rank, model_parallel_size):
    dist_init(rank, model_parallel_size)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing initialize_affine_weight with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size, output_size_coeff, 0, torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff, dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print(
        "   column parallel max error (should be zero) on global rank {}: {}".format(
            torch.distributed.get_rank(), error
        )
    )
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size, input_size_coeff, 1, torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff, dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print(
        "   row parallel max error (should be zero) on global rank {}: {}".format(torch.distributed.get_rank(), error)
    )
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def run_test_column_parallel_linear(rank, model_parallel_size):
    dist_init(rank, model_parallel_size)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    my_dLdb = torch.split(dLdb, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


def run_test_row_parallel_linear(rank, model_parallel_size):
    dist_init(rank, model_parallel_size)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing RowParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.RowParallelLinear(input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff, dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


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


def run_test_pipe(rank, world_size):
    pipe_world_size = 2

    if world_size == 1:
        return

    listen()
    dist_init(rank, world_size)
    import os

    os.environ["MASTER_PORT"] = "29502"
    # rpc.init_rpc(f"Test{rank}", backend=rpc.BackendType.TENSORPIPE, rank=rank, world_size=world_size)
    rpc.init_rpc(f"Test{rank}", rank=rank, world_size=world_size)

    mpu.initialize_model_parallel(world_size / pipe_world_size, pipe_world_size)
    model_parallel_size = mpu.get_model_parallel_world_size()
    if torch.distributed.get_rank() == 0:
        print(
            "> testing Sequential + Pipe with model parallel size: {}, pipe: {}".format(
                model_parallel_size, pipe_world_size
            )
        )
    chunk_size = 4

    def autograd_back_hook(module, grad_input, grad_output):
        print(f"hooked output {torch.distributed.get_rank()}, {module}\nin: {grad_input}\nout: {grad_output}")
        import traceback

        traceback.print_stack()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 3
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 7
    output_size = output_size_coeff * model_parallel_size
    batch_size = 3 * chunk_size

    target = torch.rand((batch_size, input_size), requires_grad=True).cuda()
    print(f"target = {target}")

    identity = IdentityLayer2D(batch_size, input_size).cuda()

    pipeline_devices = mpu.get_pipeline_parallel_group()

    set_random_seed(seed)
    model = nn.Sequential(
        layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True, bias=False).cuda(),
        nn.ReLU(),
        layers.RowParallelLinear(output_size, input_size, keep_master_weight_for_test=True, bias=False).cuda(),
    )
    for submodule in model:
        submodule.register_backward_hook(autograd_back_hook)

    set_random_seed(seed)

    reference = [
        nn.Linear(input_size, output_size, bias=False).cuda(),
        nn.ReLU(),
        nn.Linear(output_size, input_size, bias=False).cuda(),
    ]
    for submodule in reference:
        submodule.register_backward_hook(autograd_back_hook)

    reference[0].weight = Parameter(model[0].master_weight.clone())
    reference[2].weight = Parameter(model[2].master_weight.clone())

    reference = nn.Sequential(*reference)

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

    def check_weights(x, y, key: str, index=None):
        for i in [2, 0]:
            if index is not None and i != index:
                continue
            left = gather_from_model_parallel_region(x[i].weight.data)
            right = y[i].weight.data
            if not torch.allclose(left, right, atol=1.0e-6) or index is not None:
                print(f"check_weights {key}-{i}: left = {left}, \nright = {right}")
            if not torch.equal(left, right):
                print(f"check_weights NOT_EQUAL {key}-{i}: left = {left}, \nright = {right}")
            assert torch.allclose(left, right, atol=1.0e-6)

    def dump_opt_params(opt):
        for i, group in enumerate(opt.param_groups):
            for j, p in enumerate(group["params"]):
                print(f"{torch.distributed.get_rank()}:param {(i,j)} = {p}")
                print(f"{torch.distributed.get_rank()}:param.grad {(i,j)} = {p.grad}")

    def forward_model(model_, target, step=False):
        optimizer = torch.optim.SGD(model_.parameters(), lr=0.01, momentum=0.9)
        optimizer.zero_grad()
        model_.zero_grad()
        output = model_(identity())
        dump_grad(output)
        loss = nn.MSELoss()
        model_.zero_grad()
        if step:
            loss(output, target).backward()
            saved_weight_0 = model_[0].weight.data.clone()
            saved_weight_2 = model_[2].weight.data.clone()
            # print(f"model grad = {model_[0].weight.grad}")
            dump_opt_params(optimizer)
            optimizer.step()
            assert not torch.allclose(saved_weight_0, model_[0].weight.data, atol=1.0e-6)
            assert not torch.allclose(saved_weight_2, model_[2].weight.data, atol=1.0e-6)
        return output

    # print(f"ref weight[0] {reference[0].weight.data}")
    # print(f"ref weight[2] {reference[2].weight.data}")

    output = forward_model(model, target)
    reference_output = forward_model(reference, target)
    # print(f"ref_out= {reference_output}, {target}")

    error = reference_output.sub(output).max()
    torch.distributed.barrier()
    assert error < 1.0e-6

    output = forward_model(model, target)
    dump_grad(output, "mp_model")
    error = reference_output.sub(output).max()
    torch.distributed.barrier()
    assert error < 1.0e-6

    output = forward_model(model, target)
    error = reference_output.sub(output).max()
    torch.distributed.barrier()
    assert error < 1.0e-6

    saved_weight_0 = model[0].weight.data.clone()
    saved_weight_2 = model[2].weight.data.clone()
    output = forward_model(model, target, step=True)
    error = reference_output.sub(output).max()
    assert error < 1.0e-6
    # check_weights(model, reference, "before")
    model[0].weight.data = saved_weight_0
    model[2].weight.data = saved_weight_2

    if pipe_world_size == 2:
        assert torch.equal(saved_weight_0, model[0].weight.data)
        assert torch.equal(saved_weight_2, model[2].weight.data)
        pipe_model = Pipe(model, [2, 1], devices=pipeline_devices, chunks=chunk_size)
        torch.distributed.barrier()
        if rank == 0:
            assert torch.equal(saved_weight_0, pipe_model[0].weight.data)
        else:
            assert torch.equal(saved_weight_2, pipe_model[0].weight.data)
        optimizer = torch.optim.SGD(pipe_model.parameters(), lr=0.01, momentum=0.9)
        optimizer.zero_grad()
        if rank == 0:
            assert torch.equal(saved_weight_0, pipe_model[0].weight.data)
            print(f"runner {rank}:\n{pipe_model[0].weight.data}")
        else:
            assert torch.equal(saved_weight_2, pipe_model[0].weight.data)
            print(f"runner {rank}:\n{pipe_model[0].weight.data}")

        pipe_output = pipe_model(identity())
        print(f"exited pipe for {rank}")
        forward_model(reference, target, step=True)

        if torch.distributed.get_rank(mpu.get_pipeline_parallel_group()) == 1:
            error = reference_output.sub(pipe_output.cuda()).max()
            assert error < 1.0e-6

            print(f"pipe_output = {pipe_output}")
            dump_grad(pipe_output, "pipe_model")
            loss = nn.MSELoss()
            loss(pipe_output, target).backward()
            dump_opt_params(optimizer)
            optimizer.step()


            print(f"calling check_weights on master")
            check_weights(model, reference, "pipe", index=2)
            print(f"waiting for barrier on master, pid={os.getpid()}")
        else:
            print(f"calling backwards on slave, pid={os.getpid()}")
            pipe_model.back_helper(pipe_output)
            dump_opt_params(optimizer)
            print(f"calling step on slave")
            optimizer.step()
            print(f"calling check_weights on slave")
            check_weights(model, reference, "pipe", index=0)
            print(f"waiting for barrier on slave")

        pipe_model.zero_grad()
        torch.distributed.barrier()

        pipe_output = pipe_model(identity())
        updated_ref_output = forward_model(reference, target)
        if torch.distributed.get_rank(mpu.get_pipeline_parallel_group()) == 1:
            error = updated_ref_output.sub(pipe_output.cuda()).max()
            print(f"outputs are ref:\n{updated_ref_output}\npipe:\n{pipe_output}")
            assert error < 1.0e-6
        torch.distributed.barrier()

        print(f"finished waiting for barrier on, pid={os.getpid()}")

    print(f"really exited pipe for {rank}")

    rpc.shutdown()


def print_globals():
    globals()["run_test_tp"].yatta = "hi"
    globals()["run_test_tp"].event.set()


def run_test_tp(rank):
    import threading
    import os

    event = threading.Event()
    run_test_tp.event = event
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    # rpc.init_rpc(f"Test{rank}", backend=rpc.BackendType.TENSORPIPE, rank=rank, world_size=2)

    if rank == 0:

        rpc.rpc_sync("Test1", print_globals)
        event.set()

        event.wait()
        assert hasattr(run_test_tp, "yatta") == False

    elif rank == 1:
        event.wait()
        assert run_test_tp.yatta == "hi"

    rpc.shutdown()


def test_tp():
    import torch.multiprocessing as mp

    mp.spawn(run_test_tp, nprocs=2, join=True)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_affine_weight():
    spawn_for_all_world_sizes(run_test_initialize_affine_weight)


def test_embedding():
    spawn_for_all_world_sizes(run_test_parallel_embedding)


def test_column_parallel():
    spawn_for_all_world_sizes(run_test_column_parallel_linear)


def test_row_parallel():
    spawn_for_all_world_sizes(run_test_row_parallel_linear)


def test_pipe_layer():
    world_sizes = [x for x in get_world_sizes() if x <= torch.cuda.device_count() / 2]

    spawn_for_all_world_sizes(run_test_pipe)  # , world_sizes)
