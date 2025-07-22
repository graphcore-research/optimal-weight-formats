import os
import sys

import torch

if __name__ == "__main__":
    print(os.environ)
    torch.distributed.init_process_group()
    try:
        device = torch.device("cuda", torch.distributed.get_rank())
        torch.set_default_device(device)
        print("pre all_reduce", file=sys.stderr, flush=True)
        t = torch.tensor(float(torch.distributed.get_rank()))
        torch.distributed.all_reduce(t)
        print("result", t, file=sys.stderr, flush=True)
    finally:
        torch.distributed.destroy_process_group()
