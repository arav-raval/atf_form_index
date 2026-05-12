# Top-K Retrieval

Simulated query: a user types the *true* serial; the system ranks all
extracted-serial hypotheses by confusion-aware edit distance and returns
the top-K candidate documents. P(top-K) = fraction of queries where the
true source document appears within the top-K.

| K | P(top-K) |
|---:|---:|
| 1 | 0.6269 (62.69%) |
| 3 | 0.6396 (63.96%) |
| 5 | 0.6438 (64.38%) |
| 10 | 0.6472 (64.72%) |
| 25 | 0.6557 (65.57%) |
| 50 | 0.6650 (66.50%) |
| 100 | 0.6844 (68.44%) |
