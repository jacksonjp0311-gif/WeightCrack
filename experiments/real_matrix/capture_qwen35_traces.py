"""Extract Qwen3.5-0.8B layer 15 down_proj and capture math-loop traces.

The loop rule is unchanged: a 4-word sequence occurs at least three times
in a greedy completion of at least 12 words. Holdout prompts are listed and
are not formatted, tokenized, or generated. A failed download or load is
appended to docs/REAL_MATRIX_STATUS.md and the process stops.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightcrack.canonical import array_content_sha256, file_sha256

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LAYER_INDEX = 15
CACHE = ROOT / ".venv-real" / "hf-cache"
HERE = ROOT / "experiments" / "real_matrix"
OUT = HERE / "mlp_down_proj.npz"
META = HERE / "matrix_meta.json"
TRACES = HERE / "traces.npz"
LABELS = HERE / "trace_labels.json"
STATUS = ROOT / "docs" / "REAL_MATRIX_STATUS.md"
TARGET_FAIL = 10
MAX_NEW_TOKENS = 384

PRACTICE = [
    {"id": "p01", "prompt": "Solve: 2 + 2 ="},
    {"id": "p02", "prompt": "What is 17 * 23? Give the final number."},
    {"id": "p03", "prompt": "Compute 144 / 12."},
    {"id": "p04", "prompt": "What is the sum of the integers from 1 through 20? Show the work."},
    {"id": "p05", "prompt": "Solve 3x + 5 = 26. Show each step and keep checking the arithmetic."},
    {"id": "p06", "prompt": "What is 2 to the 10th power?"},
    {"id": "p07", "prompt": "A rectangle is 7 by 9. What is its area?"},
    {"id": "p08", "prompt": "What is 15 percent of 80? Explain the calculation in detail and verify it."},
    {"id": "p09", "prompt": "List the prime numbers between 10 and 30 and count them. Check the list again."},
    {"id": "p10", "prompt": "What is the square root of 81?"},
    {"id": "p11", "prompt": "A train goes 60 miles in 1.5 hours. What is its average speed? Derive it step by step and repeat the check until the units match."},
    {"id": "p12", "prompt": "What is 999 + 1?"},
    {"id": "p13", "prompt": "Compute 8 factorial and show the product."},
    {"id": "p14", "prompt": "What is 7 squared plus 3 squared?"},
    {"id": "p15", "prompt": "Convert 3/4 to a decimal and prove the division."},
    {"id": "p16", "prompt": "What is the remainder when 100 is divided by 7? Show the division."},
    {"id": "p17", "prompt": "Solve 9x = 72."},
    {"id": "p18", "prompt": "What is 12 * 13? Write the multiplication and then confirm it a second time."},
    {"id": "q01", "prompt": "What is 20 + 22?"},
    {"id": "q02", "prompt": "A circle has radius 5. What is its diameter?"},
    {"id": "q03", "prompt": "What is 9 * 9?"},
    {"id": "q04", "prompt": "How many hours are in 2 days?"},
    {"id": "q05", "prompt": "Solve: 8 - 3 ="},
    {"id": "q06", "prompt": "What is 50 percent of 20?"},
]

# Sealed. This script does not format or tokenize these.
HOLDOUT = [
    {"id": "h01", "prompt": "Solve: 5 + 5 ="},
    {"id": "h02", "prompt": "What is 13 * 17?"},
    {"id": "h03", "prompt": "Solve 4x - 3 = 21."},
    {"id": "h04", "prompt": "What is 25 percent of 64?"},
    {"id": "h05", "prompt": "A triangle has base 10 and height 6. What is its area?"},
    {"id": "h06", "prompt": "What is 2 to the 8th power?"},
]


def stop(exc: BaseException) -> None:
    block = (
        "\n\n## Qwen3.5-0.8B attempt, 2026-11-10\n\n"
        f"Model: {MODEL_ID} revision {REVISION}.\n\n"
        "Result: DOWNLOAD_OR_LOAD_FAILED.\n\n"
        "```\n"
        f"{exc.__class__.__name__}: {exc}\n"
        "```\n\n"
        "```\n"
        f"{traceback.format_exc()}\n"
        "```\n\n"
        "No replacement matrix was saved by this attempt. The run stopped.\n"
    )
    previous = STATUS.read_text(encoding="utf-8") if STATUS.exists() else ""
    STATUS.write_text(previous + block, encoding="utf-8")
    print(f"DOWNLOAD_OR_LOAD_FAILED {exc.__class__.__name__}: {exc}", flush=True)
    raise SystemExit(1) from exc


def is_loop(text: str) -> bool:
    words = text.split()
    if len(words) < 12:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - 3):
        gram = tuple(words[index:index + 4])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= 3:
            return True
    return False


def layer15_key(folder: Path) -> tuple[Path, str]:
    from safetensors import safe_open

    wanted = f"layers.{LAYER_INDEX}.mlp.down_proj.weight"
    found: list[tuple[Path, str]] = []
    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(wanted) and "visual" not in key:
                    found.append((path, key))
    language = [(path, key) for path, key in found if "language_model" in key]
    chosen = language or found
    if len(chosen) != 1:
        raise RuntimeError(f"expected one layer-15 down_proj, found {chosen}")
    return chosen[0]


def read_matrix(path: Path, key: str) -> tuple[np.ndarray, str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        tensor = handle.get_tensor(key)
        original_dtype = str(tensor.dtype).replace("torch.", "")
        matrix = tensor.detach().float().contiguous().cpu().numpy()
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise RuntimeError("host cast did not produce a float32 matrix")
    return matrix, original_dtype


def load_tokenizer(folder: Path):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(folder)
    except Exception:
        processor = AutoProcessor.from_pretrained(folder)
        tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def map_text_key(key: str) -> str | None:
    if "visual" in key or key.startswith("mtp"):
        return None
    prefix = "model.language_model."
    if key.startswith(prefix):
        return "model." + key[len(prefix):]
    if key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.")):
        return key
    return None


def load_model(folder: Path, matrix: np.ndarray):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = AutoConfig.from_pretrained(folder)
    text_config = config.text_config if hasattr(config, "text_config") else config
    if isinstance(text_config, dict):
        text_config = Qwen3_5TextConfig(**text_config)
    model = Qwen3_5ForCausalLM(text_config).to(dtype=torch.float32)
    loaded: set[str] = set()
    for path in sorted(folder.glob("*.safetensors")):
        shard = load_file(str(path))
        mapped = {}
        for key, value in shard.items():
            renamed = map_text_key(key)
            if renamed is None:
                continue
            mapped[renamed] = value.detach().float().contiguous()
        model.load_state_dict(mapped, strict=False)
        loaded.update(mapped)
        del shard, mapped
    missing = [key for key in model.state_dict() if key not in loaded]
    if missing == ["lm_head.weight"]:
        model.lm_head.weight = model.model.embed_tokens.weight
        missing = []
    if missing:
        raise RuntimeError(f"text weights missing after load: {missing[:8]}")
    model.eval()
    module = model.model.layers[LAYER_INDEX].mlp.down_proj
    if tuple(module.weight.shape) != tuple(matrix.shape):
        raise RuntimeError(
            f"layer {LAYER_INDEX} shape {tuple(module.weight.shape)} does not match stored matrix {tuple(matrix.shape)}"
        )
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(np.array(matrix, dtype=np.float32, copy=True)))
    return model, module


def trace_last(model, module, input_ids):
    import torch

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["y"] = tensor.detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["y"][0, -1].numpy().astype(np.float32)


def generate(model, tokenizer, rendered: str) -> str:
    import torch

    encoded = tokenizer(rendered, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def main() -> int:
    prompts = [item["prompt"] for item in PRACTICE]
    if len(prompts) != len(set(prompts)):
        raise SystemExit("practice prompts are not unique")
    for held in HOLDOUT:
        if held["prompt"] in prompts:
            raise SystemExit("holdout prompt leaked into practice")
    try:
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(CACHE),
        ))
        weight_path, key = layer15_key(folder)
        matrix, original_dtype = read_matrix(weight_path, key)
        tokenizer = load_tokenizer(folder)
        sample = render_prompt(tokenizer, PRACTICE[0]["prompt"])
        model, module = load_model(folder, matrix)
    except SystemExit:
        raise
    except Exception as exc:
        stop(exc)
    rows = []
    fail = []
    success = []
    done = set()
    if LABELS.is_file() and TRACES.is_file():
        previous = json.loads(LABELS.read_text(encoding="utf-8"))
        if previous.get("model_id") == MODEL_ID and previous.get("revision") == REVISION:
            saved = np.load(TRACES)
            fail_rows = list(saved["fail"]) if "fail" in saved.files else []
            success_rows = list(saved["success"]) if "success" in saved.files else []
            fail_index = 0
            success_index = 0
            for row in previous["practice"]:
                rows.append(row)
                done.add(row["id"])
                if row["loop"]:
                    fail.append(fail_rows[fail_index])
                    fail_index += 1
                else:
                    success.append(success_rows[success_index])
                    success_index += 1
            if fail_index != len(fail_rows) or success_index != len(success_rows):
                raise SystemExit("saved traces do not match the label rows")
            print(f"resume fail={len(fail)} success={len(success)}", flush=True)
    for item in PRACTICE:
        if item["id"] in done:
            continue
        rendered = render_prompt(tokenizer, item["prompt"])
        encoded = tokenizer(rendered, return_tensors="pt")
        vector = trace_last(model, module, encoded["input_ids"])
        text = generate(model, tokenizer, rendered)
        looping = is_loop(text)
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "loop": looping,
            "completion": text,
        })
        if looping:
            fail.append(vector)
        else:
            success.append(vector)
        print(f"{item['id']} loop={looping} fail={len(fail)}", flush=True)
        if len(fail) >= TARGET_FAIL:
            break
    if matrix.shape[0] != (fail[0].shape[0] if fail else success[0].shape[0]):
        raise SystemExit("trace width does not match the matrix rows")
    np.savez(OUT, weight=matrix)
    meta = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_file_sha256": file_sha256(weight_path),
        "source_key": key,
        "layer_index": LAYER_INDEX,
        "checkpoint_dtype": original_dtype,
        "stored_dtype": "float32",
        "shape": list(matrix.shape),
        "content_sha256": array_content_sha256(matrix),
        "npz": str(OUT),
        "npz_key": "weight",
        "chat_template_sample_prefix": sample[:240],
    }
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    payload = {}
    if fail:
        payload["fail"] = np.stack(fail).astype(np.float32)
    if success:
        payload["success"] = np.stack(success).astype(np.float32)
    if payload:
        np.savez(TRACES, **payload)
    status = "captured" if len(fail) >= 8 else "below_density_target"
    LABELS.write_text(json.dumps({
        "status": status,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source_key": key,
        "loop_rule": "a 4-word sequence occurs at least 3 times",
        "max_new_tokens": MAX_NEW_TOKENS,
        "target_fail": TARGET_FAIL,
        "fail_count": len(fail),
        "success_count": len(success),
        "prompts_run": len(rows),
        "practice": rows,
        "holdout_opened": False,
        "holdout_ids": [item["id"] for item in HOLDOUT],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "fail": len(fail),
        "success": len(success),
        "shape": list(matrix.shape),
        "key": key,
    }, indent=2))
    return 0 if len(fail) >= 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
