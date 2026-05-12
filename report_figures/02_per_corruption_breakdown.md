# Per-Corruption-Type Breakdown

Verifier threshold: **0.95**

| Corruption type | Category | n | Admitted | Leak% |
|---|---|---:|---:|---:|
| `pii_in_serial` | PII | 432 | 21 | 4.86% |
| `name_in_serial` | PII | 449 | 1 | 0.22% |
| `serial_overflow` | Positive (correct admit) | 411 | 181 | 44.04% |
| `overflow_into_serial` | Data-Quality | 414 | 105 | 25.36% |
| `field_swap` | Data-Quality | 462 | 22 | 4.76% |

**Categories:**
- **PII**: privacy-violating — must be rejected for compliance.
- **Data-quality**: wrong data (field swap or overflow from neighbor cell). Not personal info, but should not enter the index.
- **Positive**: `serial_overflow` is labeled positive in our scheme because the serial IS in the box, just messy. High admit rate is correct.
