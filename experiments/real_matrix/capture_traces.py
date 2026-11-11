"""Capture practice math-loop traces at the extracted down-projection.

Holdout prompts are listed and are not generated. A loop is a 4-word
sequence that occurs at least three times in the greedy completion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "mlp_down_proj.npz"
META = HERE / "matrix_meta.json"
TRACES = HERE / "traces.npz"
LABELS = HERE / "trace_labels.json"
CACHE = ROOT / ".venv-real" / "hf-cache"

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
]

# Sealed. capture_traces does not format or tokenize these.
HOLDOUT = [
    {"id": "h01", "prompt": "Solve: 5 + 5 ="},
    {"id": "h02", "prompt": "What is 13 * 17?"},
    {"id": "h03", "prompt": "Solve 4x - 3 = 21."},
    {"id": "h04", "prompt": "What is 25 percent of 64?"},
    {"id": "h05", "prompt": "A triangle has base 10 and height 6. What is its area?"},
    {"id": "h06", "prompt": "What is 2 to the 8th power?"},
]


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


def load_model(matrix: np.ndarray, layer_index: int, model_id: str, revision: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=str(CACHE))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(CACHE),
        dtype=torch.float32,
    )
    model.eval()
    module = model.model.layers[layer_index].mlp.down_proj
    if tuple(module.weight.shape) != tuple(matrix.shape):
        raise RuntimeError(
            f"module shape {tuple(module.weight.shape)} does not match stored matrix {tuple(matrix.shape)}"
        )
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(np.array(matrix, dtype=np.float32, copy=True)))
    return model, tokenizer, module


def trace_last(model, module, input_ids: torch.Tensor) -> np.ndarray:
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


def generate(model, tokenizer, prompt: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> int:
    if not SOURCE.is_file() or not META.is_file():
        raise SystemExit("matrix is missing; run extract_matrix.py first")
    meta = json.loads(META.read_text(encoding="utf-8"))
    matrix = np.load(SOURCE)["weight"]
    model, tokenizer, module = load_model(
        matrix,
        int(meta["layer_index"]),
        meta["model_id"],
        meta["revision"],
    )
    rows = []
    fail = []
    success = []
    for item in PRACTICE:
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        vector = trace_last(model, module, encoded["input_ids"])
        text = generate(model, tokenizer, item["prompt"])
        looping = is_loop(text)
        row = {
            "id": item["id"],
            "prompt": item["prompt"],
            "loop": looping,
            "completion": text,
        }
        rows.append(row)
        if looping:
            fail.append(vector)
        else:
            success.append(vector)
        print(f"{item['id']} loop={looping}", flush=True)
    if not fail:
        LABELS.write_text(json.dumps({
            "status": "no_loop_failures",
            "loop_rule": "a 4-word sequence occurs at least 3 times",
            "practice": rows,
            "holdout_opened": False,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "no_loop_failures", "practice": len(rows)}, indent=2))
        return 0
    payload = {"fail": np.stack(fail).astype(np.float32)}
    if success:
        payload["success"] = np.stack(success).astype(np.float32)
    np.savez(TRACES, **payload)
    LABELS.write_text(json.dumps({
        "status": "captured",
        "loop_rule": "a 4-word sequence occurs at least 3 times",
        "fail_count": len(fail),
        "success_count": len(success),
        "practice": rows,
        "holdout_opened": False,
        "holdout_ids": [item["id"] for item in HOLDOUT],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fail": len(fail), "success": len(success)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
