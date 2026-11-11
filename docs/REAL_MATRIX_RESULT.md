# Real-matrix probe result

Date: 2026-11-10. Branch `dev/verified-projection-repair`.

The host gate admitted one edit. The sealed holdout was opened once. The edit did not move held-out behavior. The success criterion is not met.

## Qwen2.5-0.5B practice expansion

The 4-word rule is unchanged: a 4-word sequence occurs at least three times, and the completion has at least 12 words. Greedy decoding, 128 new tokens. The sealed holdout prompts were not tokenized during this expansion.

`experiments/real_matrix/capture_dense_traces.py` ran 52 prompts. Two looped: `p02` and `d18`. The other 50 did not. That is below the 8-failure target, so the run switched to Qwen3.5-0.8B layer 15.

## Matrix

`Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`.

Tensor `model.language_model.layers.15.mlp.down_proj.weight`. Checkpoint dtype bfloat16. Stored array float32, shape `(1024, 3584)`, npz key `weight`. Content SHA-256 `b75ed374dddf0d5c9443f12b5d7d7af05d5234d264965911419ef3616fb0144d`. The shard that holds the tensor has SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

File: `experiments/real_matrix/mlp_down_proj.npz`. Capture: `experiments/real_matrix/capture_qwen35_traces.py`. The language-model head is tied to the token embeddings, so the loader shares `lm_head.weight` with `model.embed_tokens.weight`.

## Practice traces

Thinking was enabled. Greedy decoding, 384 new tokens. The same 4-word rule. Holdout prompts were still sealed.

24 prompts were generated. Eight met the loop rule: `p01`, `p07`, `p09`, `p10`, `p15`, `p17`, `p18`, `q05`. Sixteen did not. `experiments/real_matrix/traces.npz` contains `fail` with shape `(8, 1024)` and `success` with shape `(16, 1024)`.

These eight rows are the prompts the frozen rule marked. They are not a reproduction of a published MATH-500 token-limit rate.

## Probe

Command:

```
python -m weightcrack.evaluation.real_matrix_probe experiments/real_matrix/mlp_down_proj.npz experiments/real_probe experiments/real_matrix/traces.npz
```

`experiments/real_probe/probe_report.json`:

- status `probe_complete`
- matrix hash `b75ed374dddf0d5c9443f12b5d7d7af05d5234d264965911419ef3616fb0144d`
- failure traces 8
- selected columns 2056 and 1309
- strength 0.5
- touched coordinates 2048
- max absolute stored delta 0.003278465010225773

The candidate was built with `construct_candidate`. The host gate admitted it. `apply_patch` published `experiments/real_probe/finalist.wcbundle`. The receipt status is `applied`. Candidate content SHA-256 `6dfb1e4dd349616ccb4beea1a53f7ef15a8549f6b6a8bd92229d93bcc699e21e`. That hash was written to `experiments/real_probe/holdout_started.json` before any holdout prompt was tokenized.

## Holdout

Opened once. Six sealed prompts. Same rule, same 384-token greedy decode, thinking enabled. Baseline uses the stored matrix. The edited pass uses the published candidate.

| id | baseline loop | edited loop |
| --- | --- | --- |
| h01 | true | true |
| h02 | false | false |
| h03 | true | true |
| h04 | true | true |
| h05 | false | false |
| h06 | true | true |

Baseline loops 4 of 6. Edited loops 4 of 6. Every prompt kept its label. Held-out behavior did not move.

Details: `experiments/real_probe/holdout.json`. Receipt: `experiments/real_probe/finalist.wcbundle/receipt.json`.

## Criterion

The requested success criterion is a policy-admitted edit on a real matrix that moves held-out behavior on a dense failure mode, with the receipt written. The gate admitted the edit and the receipt is written. Held-out behavior did not move, so the criterion is not met.
