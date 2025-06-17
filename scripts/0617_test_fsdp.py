import os
from typing import Any

import torch
import torch.distributed.fsdp
import torch.multiprocessing as multiprocessing
from torch import nn


def train(s: dict[str, Any]) -> nn.Module:
    # EITHER
    # os.environ.update(s)
    # torch.distributed.init_process_group()
    # OR
    torch.distributed.init_process_group(
        world_size=int(s["WORLD_SIZE"]),
        rank=int(s["RANK"]),
        init_method=f"tcp://{s['MASTER_ADDR']}:{s['MASTER_PORT']}",
    )

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.device("cuda", rank)
        # torch.set_default_device(device)
        torch.cuda.set_device(device)
        print("device", device)

        torch.manual_seed(100)
        module = nn.Sequential(nn.Linear(10, 30), nn.Linear(30, 30)).to(device)
        input = (
            torch.randn(16, 10)
            .to(device)
            .bfloat16()
            .unflatten(0, (world_size, -1))[rank]
        )
        grad_output = (
            torch.randn(16, 30)
            .to(device)
            .bfloat16()
            .unflatten(0, (world_size, -1))[rank]
        )

        for m in module:
            torch.distributed.fsdp.fully_shard(
                m,
                mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16
                ),
            )
        opt = torch.optim.Adam(module.parameters())

        for _ in range(2):
            opt.zero_grad()
            out = module(input)
            out.backward(grad_output)
            opt.step()
            print(rank, out[0])

        print(
            rank,
            "params",
            {
                k: (v.dtype, v.to_local().shape, type(v))
                for k, v in module.named_parameters()
            },
        )

        for m in module.modules():
            if isinstance(m, torch.distributed.fsdp.FSDPModule):
                m.unshard()
        return module
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    s = dict(
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT="10002",
    )
    processes = [
        multiprocessing.Process(
            target=train, args=(dict(**s, RANK=str(n), LOCAL_RANK=str(n)),)
        )
        for n in range(1, world_size)
    ]
    for p in processes:
        p.start()
    model = train(dict(**s, RANK="0", LOCAL_RANK="0"))
    for p in processes:
        p.join()

    print("root", {k: (v.dtype, v.shape, type(v)) for k, v in model.named_parameters()})

    out = model(torch.randn(1, 10).cuda().bfloat16())
    print("root", out)

    # This should break (workers have died)
    # for m in model.modules():
    #     if isinstance(m, torch.distributed.fsdp.FSDPModule):
    #         m.reshard()
    # out = model(torch.randn(16, 10).cuda().bfloat16())
    # print("root resharded", out)
