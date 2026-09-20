# Fine-tune SmolLM2 with Unsloth

Run a small but real LoRA fine-tuning job with
[`unsloth/SmolLM2-135M-Instruct`](https://huggingface.co/unsloth/SmolLM2-135M-Instruct)
and 64 public examples from `yahma/alpaca-cleaned`. The model is about 269 MB and
the recipe trains for 10 steps, so it is useful for validating a GPU workflow without
paying for a long training run.

```bash
cd vkong-examples/smollm2-unsloth-finetuning
vkong run -C . --auto-stop
```

VKong creates the `unsloth-smollm2-lab` volume automatically and mounts it at
`/data`. The Hugging Face cache, intermediate checkpoint, final LoRA adapter, and
completion marker all stay on that volume. Once the GPU stops, VKong saves the volume.

Run the same command again to test restore on a fresh machine. The script validates the
saved adapter and prints `storage restore verified` instead of training again.

The recipe asks for 50 GB of machine disk because VKong must fit the container image,
runtime files, and restored volume on the rented machine. Durable storage is billed by
the volume's actual allocated size, separately from GPU compute.
