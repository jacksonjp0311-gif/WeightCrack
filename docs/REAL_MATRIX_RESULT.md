# Real-matrix probe result

Date: 2026-11-10. Branch `dev/verified-projection-repair`.

No candidate qualified. Holdout stayed closed. No receipt was written. This is not a policy-admitted edit, and it did not move held-out behavior.

## Matrix

`Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. The local snapshot was already present, so this run did not download.

Tensor `model.layers.23.mlp.down_proj.weight`, the highest numeric `mlp.down_proj.weight` key. Checkpoint dtype bfloat16. Stored array float32, shape `(896, 4864)`, npz key `weight`. Content SHA-256 `689bb57897d47e3d9c8b645d4d1bd5b5ef1645524fe6211683c67372a6179988`. Checkpoint file SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.

File: `experiments/real_matrix/mlp_down_proj.npz`. Extractor: `experiments/real_matrix/extract_matrix.py`.

## Practice traces

Eighteen math prompts were generated greedily with 128 new tokens. A loop is a 4-word sequence that occurs at least three times. The sealed holdout prompts in `capture_traces.py` were not tokenized.

One prompt looped: `p02`, "What is 17 * 23? Give the final number." The completion repeats the pair of sentences that multiply the ones digits and the tens digits. The other 17 prompts did not meet the loop rule. This practice set is not a dense loop failure mode.

`experiments/real_matrix/traces.npz` contains `fail` from that one looping row and `success` from the 17 others. Labels are in `experiments/real_matrix/trace_labels.json`.

## Probe

Command:

```
python -m weightcrack.evaluation.real_matrix_probe experiments/real_matrix/mlp_down_proj.npz experiments/real_probe experiments/real_matrix/traces.npz
```

`experiments/real_probe/probe_report.json`:

- status `probe_complete`
- matrix hash `689bb57897d47e3d9c8b645d4d1bd5b5ef1645524fe6211683c67372a6179988`
- selected columns 2252 and 3396
- touched coordinates 1792
- max absolute stored delta 0.005169042386114597
- note: "Structural probe on a loaded matrix. Not a language-model generation measurement."

The probe builds its direction from random normal vectors at seed 0. It does not read `traces.npz`. It calls `apply_project_remove` and does not call `construct_candidate`. It has no practice score and no zero-loss gate.

Nothing was admitted by the host gate. `finalist` was not locked. Holdout was not opened.

## Criterion

The requested success criterion is a policy-admitted edit on a real matrix that moves held-out behavior on a dense failure mode. This run does not meet it.
