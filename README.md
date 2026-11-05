# WeightCrack

Surgical weight-editing research scaffold.  
Host-gated sparse patch proposals, policy enforcement, practice/holdout separation, receipt-based audit.  
The crack-open track for localization-and-proposal under governance invariants.

## Goal

A meta-algorithm that can take any model, localize the weights responsible for a target sector (math, coding, refusal, style, etc.), propose a sparse edit, have the host gate and apply it, and measure a performance upgrade on a frozen holdout while preserving sentinel behavior.

## Current foundation

- Patch and policy schemas (`weightcrack/schema.py`)
- Host-side policy gate (`weightcrack/policy.py`)
- Adapted patch applicator stub (`weightcrack/patches/huggingface.py`) — receipt path in place; tensor-level write pending full safetensors integration
- Evaluation loop stub (`weightcrack/evaluation/loops.py`)
- Direction-trial primitive stub (`weightcrack/agents/direction_trial.py`)
- Meta-algorithm sketch (`docs/META_ALGORITHM.md`)

## Next

Fill the tensor-level apply path and the practice-only activation/direction probe so an agent can propose against real signal.
