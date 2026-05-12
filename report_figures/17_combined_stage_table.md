# End-to-End Stage Results — Combined

Verifier operating threshold: **0.95**

| Subset | Pages | Classifier acc | Positives | Admit% | OCR% | Exact% | Negatives | Leak% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Master held-out | 785 | 100.00% | 1182 | 65.5% | 65.5% | 44.0% | 0 | — |
| Compliance (all) | 2385 | 99.75% | 1109 | 54.2% | 54.2% | 28.4% | 1549 | 3.36% |
| Compliance — clean only | 202 | 100.00% | 242 | 66.9% | 66.9% | 45.5% | 0 | — |
| Compliance — corrupted only | 2183 | 99.73% | 867 | 50.6% | 50.6% | 23.6% | 1549 | 3.36% |

**Definitions** — Admit% = verifier admits a true positive; OCR% = admitted *and* OCR produced output; Exact% = OCR matched truth exactly; Leak% = negatives admitted (lower better, especially on compliance corrupted).
