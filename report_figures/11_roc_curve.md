# ROC — Verifier

Operating threshold: **0.95**

| Subset | AUC | TPR @ saved threshold | FPR @ saved threshold |
|---|---:|---:|---:|
| All corruptions | 0.8341 | 0.5419 | 0.0336 |
| PII only | 0.8187 | 0.5063 | 0.0250 |
| Data-quality only | 0.7941 | 0.5063 | 0.0449 |

Operating-point dots show the verifier's behavior at the saved
threshold; the AUC summarizes performance across all thresholds.
