# End-to-End Error Decomposition

Master held-out, **1182 positive firearms**, threshold **0.95**.

Where each positive ends up in the pipeline (mutually exclusive):

| Outcome | Count | % of positives |
|---|---:|---:|
| Exact match | 520 | 43.99% |
| OCR near-miss (best hyp CER ≤ 0.20) | 150 | 12.69% |
| OCR far-miss (CER > 0.20) | 104 | 8.80% |
| Verifier admitted, no OCR output | 0 | 0.00% |
| Verifier rejected | 408 | 34.52% |
| Classifier wrong year | 0 | 0.00% |
| **Total** | **1182** | **100.0%** |
