# Real transfer v1

Qwen/Qwen2.5-0.5B revision 060db6499f32faf8b98477b0a26969ef7d8b9987. Layer 23 mlp.down_proj.weight, float32, shape (896, 4864). Content SHA-256 689bb57897d47e3d9c8b645d4d1bd5b5ef1645524fe6211683c67372a6179988.

Practice traces: 4 success rows, 6 failure rows. Baseline exact cat-repetition score 4/10.

rank_columns selected columns 4391 and 4861. Default gate rejected (1792 coordinates exceed max_entries 64). Pre-registered transfer gate (max_entries 1792, max_fraction 0.01, max_abs_delta 1.0) admitted all seven strengths 0.05–2.0. Every trial stayed at 4/10 with 0 losses. Completion text matched baseline. Largest stored absolute delta 0.40465621650218964. Finalist null.

No practice candidate qualified. Holdout stayed closed. No receipt written. Model not improved on this task.

The instrument on these six failure residuals did not recover columns whose projection removal corrects the failure under the tested strengths.
