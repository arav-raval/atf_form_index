# Compliance Confusion Matrices

Verifier threshold: **0.95**  |  
Held-out v2 _2 + Serial Error Pages

## All corruption types combined

### All corruptions

|  | Verifier ADMIT | Verifier REJECT |
|---|---:|---:|
| Real serial | 601 (TP) | 508 (FN) |
| Negative | 52 (FP) | 1497 (TN) |

- Recall: **0.5419**
- Specificity: **0.9664**
- Precision: **0.9204**
- **Leak rate: 0.0336 (3.36%)**

### PII only

|  | Verifier ADMIT | Verifier REJECT |
|---|---:|---:|
| Real serial | 439 (TP) | 428 (FN) |
| Negative | 22 (FP) | 859 (TN) |

- Recall: **0.5063**
- Specificity: **0.9750**
- Precision: **0.9523**
- **Leak rate: 0.0250 (2.50%)**

### Data-quality only

|  | Verifier ADMIT | Verifier REJECT |
|---|---:|---:|
| Real serial | 439 (TP) | 428 (FN) |
| Negative | 30 (FP) | 638 (TN) |

- Recall: **0.5063**
- Specificity: **0.9551**
- Precision: **0.9360**
- **Leak rate: 0.0449 (4.49%)**

