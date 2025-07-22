import torch
import sys
import torch.multiprocessing as multiprocessing


def _run(i: int):
    torch.distributed.init_process_group(
        rank=i,
        world_size=2,
        init_method="tcp://localhost:15300",
    )
    try:
        device = torch.device("cuda", torch.distributed.get_rank())
        torch.set_default_device(device)
        print("pre all_reduce", file=sys.stderr, flush=True)
        t = torch.tensor(float(torch.distributed.get_rank()))
        torch.distributed.all_reduce(t)
        print("result", t, file=sys.stderr, flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    processes = []
    for n in range(2):
        p = multiprocessing.Process(target=_run, args=(n,), name=f"dist-worker-{n}")
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
