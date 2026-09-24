# Fine-tune SmolLM2 with Unsloth

Run a small but real LoRA fine-tuning job with
[`unsloth/SmolLM2-135M-Instruct`](https://huggingface.co/unsloth/SmolLM2-135M-Instruct)
and 64 public examples from `yahma/alpaca-cleaned`. The model is about 269 MB and
the recipe follows Unsloth's quick-tutorial duration of 60 training steps, making it
useful for validating a complete GPU training and checkpoint workflow.

```bash
cd vkong-examples/smollm2-unsloth-finetuning
vkong run --detach
```

VKong creates the `unsloth-smollm2-lab` volume automatically and mounts it at
`/data`. The Hugging Face cache, resumable checkpoints at steps 20, 40, and 60, final
LoRA adapter, and completion marker all stay on that volume. Once the GPU stops, VKong
saves the volume.

This recipe stays online after training to serve the download. Stop the App explicitly
when finished.

After step 60, the recipe packages only the final checkpoint, LoRA adapter, and metadata
as `smollm2-unsloth-checkpoint-60.tar.gz`. Reconnect to the detached service, then use the
private localhost URL printed by VKong to download it:

```bash
vkong attach vk_<id>
curl -O http://127.0.0.1:<port>/smollm2-unsloth-checkpoint-60.tar.gz
vkong app stop smollm2-unsloth-finetuning
```

Keep `vkong attach` running while `curl` downloads the archive. The tunnel is private and
the service exposes only the archive directory, not the whole volume. Stop the App after
the download so VKong performs the controlled final save and releases the GPU.

Run the same command again to test restore on a fresh machine. The script validates the
saved adapter and prints `storage restore verified` instead of training again.

The recipe asks for 82 GB of machine disk because VKong must fit the container image,
runtime files, and restored volume on the rented machine. Durable storage is billed by
the volume's actual allocated size, separately from GPU compute.
