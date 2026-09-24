from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tarfile

import unsloth  # Patch the training stack before importing TRL or Transformers.
import torch
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


MODEL = "unsloth/SmolLM2-135M-Instruct"
OUTPUT = Path("/data/smollm2-unsloth-60-step")
CHECKPOINTS = OUTPUT / "checkpoints"
FINAL_ADAPTER = OUTPUT / "final-adapter"
MARKER = OUTPUT / "training-complete.json"
DOWNLOAD = OUTPUT / "download" / "smollm2-unsloth-checkpoint-60.tar.gz"
MAX_SEQUENCE_LENGTH = 512
MAX_STEPS = 60


def restored_run() -> bool:
    if not MARKER.is_file():
        return False
    metadata = json.loads(MARKER.read_text())
    final_checkpoint = CHECKPOINTS / f"checkpoint-{MAX_STEPS}"
    required = [
        FINAL_ADAPTER / "adapter_config.json",
        FINAL_ADAPTER / "adapter_model.safetensors",
        final_checkpoint / "trainer_state.json",
        final_checkpoint / "adapter_model.safetensors",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("saved training marker exists but the final checkpoint is incomplete")
    if metadata.get("steps") != MAX_STEPS:
        raise RuntimeError("saved training marker has an unexpected step count")
    print(
        f"storage restore verified: model={metadata['model']} "
        f"steps={metadata['steps']} adapter={FINAL_ADAPTER}",
        flush=True,
    )
    return True


def prepare_download() -> None:
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOWNLOAD.with_name(f".{DOWNLOAD.name}.tmp")
    final_checkpoint = CHECKPOINTS / f"checkpoint-{MAX_STEPS}"
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(final_checkpoint, arcname=final_checkpoint.name)
        archive.add(FINAL_ADAPTER, arcname=FINAL_ADAPTER.name)
        archive.add(MARKER, arcname=MARKER.name)
    temporary.replace(DOWNLOAD)
    print(f"checkpoint archive ready: {DOWNLOAD}", flush=True)


def serve_download() -> None:
    class DownloadHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DOWNLOAD.parent), **kwargs)

    server = ThreadingHTTPServer(("0.0.0.0", 8000), DownloadHandler)
    print(f"private download ready: /{DOWNLOAD.name}", flush=True)
    server.serve_forever()


if restored_run():
    prepare_download()
    serve_download()

OUTPUT.mkdir(parents=True, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dtype=None,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:64]")


def format_examples(batch):
    texts = []
    for instruction, context, response in zip(batch["instruction"], batch["input"], batch["output"]):
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}


dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQUENCE_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        output_dir=str(CHECKPOINTS),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        # Let the installed Unsloth version choose its compatible tokenizer
        # workers. Forcing one worker can hit a patched-config pickle failure.
        dataset_num_proc=None,
        packing=False,
    ),
)
resume_from = get_last_checkpoint(str(CHECKPOINTS)) if CHECKPOINTS.is_dir() else None
if resume_from:
    print(f"resuming training from {resume_from}", flush=True)
result = trainer.train(resume_from_checkpoint=resume_from)
final_checkpoint = CHECKPOINTS / f"checkpoint-{int(result.global_step)}"
if int(result.global_step) != MAX_STEPS or not (final_checkpoint / "trainer_state.json").is_file():
    raise RuntimeError("training ended without the expected final checkpoint")
model.save_pretrained(FINAL_ADAPTER)
tokenizer.save_pretrained(FINAL_ADAPTER)
MARKER.write_text(
    json.dumps(
        {
            "model": MODEL,
            "dataset": "yahma/alpaca-cleaned",
            "examples": len(dataset),
            "steps": int(result.global_step),
            "checkpoint": str(final_checkpoint),
            "train_loss": float(result.training_loss),
            "cuda": torch.cuda.get_device_name(0),
        },
        indent=2,
    )
    + "\n"
)
print(
    f"training complete: checkpoint={final_checkpoint} adapter={FINAL_ADAPTER}",
    flush=True,
)
prepare_download()
serve_download()
