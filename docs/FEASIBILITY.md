# Feasibility intervals on the final-layer proxy

## Storage match

Seeds 300–319. Static ranking on the top PCA direction of failure logits. Analytic strength search for a zero-loss interval in (0, 2].

Every strength the calculation admitted produced a qualifying float32 candidate after apply_project_remove.

14 feasible, 14 matched after storage.

## Observability

For an injected projection fault of strength f:

u^T w_damaged = (1-f) u^T w

At f=6 the damaged component is amplified. Magnitude ranking recovers both faulty columns in 60/60 cases. At f=1 the component is deleted and the same ranking recovers 0/60. Higher success on dense faults is partly this effect.

## Status

Analytic feasibility and stored outcome agree on the seeds tested. The calculation is exact for the final-layer classifier. It becomes an approximation for internal layers whose downstream activations change. The host gate and real-matrix transfer remain unmeasured.
