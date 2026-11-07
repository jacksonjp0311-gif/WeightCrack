## Real-matrix transfer attempt

Attempted to download Qwen/Qwen2.5-0.5B model.safetensors and extract one mlp.down_proj.weight.

Result: DOWNLOAD_FAILED (Errno 107, transport endpoint not connected). No matrix was loaded. No probe was run on real weights.

The probe harness is in weightcrack/evaluation/real_matrix_probe.py. It accepts a local .npz containing a single 'weight' array and runs the same practice-only path used in the synthetic experiment. It is ready to run when a matrix is supplied.

This is a completed attempt with an environment blocker, not a measurement of transfer.
