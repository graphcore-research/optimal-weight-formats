import contextlib
import copy
import itertools as it
from pathlib import Path

import datasets
import torch
import transformers
from torch import Tensor, nn

import weight_formats.experiments as E


def run():
    batch_size = 8
    sequence_length = 1024
    steps = 100
    model_id = "meta-llama/Llama-3.2-1B"
    dtype = torch.float16
    activation_checkpointing = False
    memory_profile = None  # Path("out/memory/tmp.pickle")
    device = torch.device("cuda")

    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        reference_model.config._name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    data = datasets.load_dataset("DKYoon/SlimPajama-6B")

    model = copy.deepcopy(reference_model)
    for p in model.parameters():
        if p.ndim == 2:
            p.data += torch.randn_like(p).mul(p.std() * 0.1)

    opt = torch.optim.Adam(model.parameters(), lr=2**-18, eps=1e-5)

    with contextlib.ExitStack() as exit:
        if activation_checkpointing:
            exit.enter_context(E.fisher.activation_checkpointing_enabled(model))
        if memory_profile:
            exit.enter_context(E.core.cuda_memory_history(memory_profile))

        for step, batch_data in enumerate(
            it.islice(data["train"].iter(batch_size=batch_size), steps)
        ):
            batch = tokenizer.batch_encode_plus(
                batch_data["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=sequence_length,
                truncation=True,
            ).to(device)
            opt.zero_grad()
            loss = E.qat.compute_kl_loss(model, reference_model, batch)
            loss.backward()
            opt.step()
            print(f"{step:>03}: {loss / batch.attention_mask.sum():.3f}")


run()
