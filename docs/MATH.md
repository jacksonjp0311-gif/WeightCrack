# Projection math and what was measured

## Implemented operation

For a matrix `W`, a direction `d` with nonzero finite norm, columns `C`, and strength `s` in `(0, 2]`:

```
u = d / ||d||
W'[:, c] = W[:, c] - s * u * (u^T W[:, c])    for c in C
```

The code computes this in float64 and casts once to the source dtype. Untouched columns keep their stored values. `s = 1` removes the component along `u`. For `s > 1` the stored component is `(1 - s)` times the original component, which reverses it. The interval `(0, 2]` is not pure removal.

Touched coordinates are `rows * |C|` after duplicate columns are rejected. The fraction denominator is `W.size`, the edited tensor only. `max_abs_delta` compares stored source entries with stored candidate entries.

## Implemented direction

On the proxy study the activation site is `post_linear_logits`: the class scores `W tanh(Ax + b) + c` of the damaged model. The direction is

```
normalize(mean(logits | practice target wrong) - mean(logits | practice target correct))
```

That is one normalized mean difference. `activation_difference` accepts `top_k` and does not use it to build further components. There is no SVD in that function. If either practice set is empty, or the difference norm is zero or nonfinite, the status is `NO_LOCALIZATION_SIGNAL` and the task is not regenerated.

Column ranking scores `|u^T W[:, c]|`. Taking the top two columns maximizes projection-removal magnitude at a fixed strength. It is not a causal attribution. The proxy comparison uses that ranking against a uniform random pair and the same `u`. It tests column selection, not direction discovery.

## Preservation-memorization

The least-squares objective that preserves some keys and rewrites others is background from the locate-then-edit literature. This repository does not implement or validate that solver.

## WeightCrack measurement

`experiments/proxy_fault_v1` is one frozen synthetic fault on `W` of an 8-16-4 classifier. The injected columns were `[10, 7]`. The heuristic selected `[13, 4]`. The random control selected `[12, 13]`.

Practice targets: 10 incorrect and 118 correct, out of 128. The heuristic finalist, dose index 2, had 1 practice correction, 0 practice losses, and 0 preservation flips. The random arm had no practice-qualified dose (`NO_QUALIFIED_CANDIDATE`).

Held-out, after the lock: the damaged model was correct on 238 of 256 target inputs. The heuristic finalist made 2 corrections and 1 loss (239 correct) and flipped 3 of 256 preservation predictions. `behavioral_repair` is false. The no-edit control matched the damaged model exactly. Wall time was 0.9394292000215501 seconds. Value reconstruction of the 16 applied bundles passed. See `docs/PROXY_FAULT_V1.md`.

This is bounded recovery evidence for one injected fault, and the repair rule failed. It is not a language-model result.

## Separate claims

These are not WeightCrack measurements. They are cited from WeightBridge commit `baa336e7c517139876a1a5d6309af7199ea1e661`. That commit is not the default main branch, which does not contain the later research narrative.

- Direct proposal, `docs/AGENT_TOOLS.md`: a pinned local Qwen controller, 20 blinded patch-form tasks on synthetic ToyMLP targets, zero finalized candidates, zero repairs.
- Phase P, `docs/PHASE_III_P.md`: reconstructed published edit, 3/48 target presentations to 8/48, six corrections, one regression, 24/24 non-target controls retained. Zero-regression qualification failed. The note says this is not a new WeightBridge learning result.
- Phase Q, `docs/PHASE_III_Q_BASELINE.md`: supervised LoRA, `VERIFIED_BASELINE_PASS`, rank 8, 5,411,328 parameters. Not a column search.
- Phase R, `docs/PHASE_III_R_REPLICATION.md`: rank-eight `VERIFIED_REPLICATION_PASS`. Rank-four `TRANSFER_WITH_PRESERVATION_FAILURE`.

Broader math and instruction numbers published by other authors are their results.
