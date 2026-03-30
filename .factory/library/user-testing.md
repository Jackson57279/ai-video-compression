# VideoQuant User Testing Guide

## Validation Concurrency

**Testing Surface:** Python Unit Tests (pytest)

**Max Concurrent Validators:** 4

The core-algorithms milestone validation runs unit tests using pytest. Tests are CPU-bound and memory-intensive during tensor operations. Each validator should be assigned a specific assertion group to test.

**Resource Cost Classification:**
- CPU-only validation: 1 concurrent validator per assertion group
- Memory usage: ~2-4GB per validator during tensor operations
- Tensor dimensions tested: [B=2, F=8-16, N=16-64, C=16-512]

## Testing Tools

**Primary Tool:** pytest (Python testing framework)

**Built-in Skills:**
- No browser automation needed - pure Python unit tests
- Use direct pytest execution via shell commands

**Test Categories:**
- TPQ tests: `tests/test_tpq.py`
- SQJL tests: `tests/test_sqjl.py`
- MAMP tests: `tests/test_mamp.py`
- Pipeline tests: `tests/test_pipeline.py`

## Setup Requirements

**Environment:**
```bash
# Virtual environment already exists at ./venv
source venv/bin/activate  # or venv/bin/python directly

# Dependencies installed via pip install -e .
```

**No Services Required:**
This is a library-only milestone with no external services (databases, APIs, etc.).

## Flow Validator Guidance: python-unit-tests

### Isolation Rules

1. **Test Isolation:** Each validator runs a distinct subset of tests
2. **No Shared State:** Tests use deterministic seeds (torch.manual_seed(42)) for reproducibility
3. **File System:** No file writes during testing (except evidence output)
4. **Memory:** Tests clean up tensors automatically via Python GC

### Boundary Constraints

- Do NOT run tests for assertions outside your assigned group
- Do NOT modify source code during testing
- Do NOT commit test results - only write to designated output paths

### Evidence Collection

For each assertion tested:
1. Run relevant pytest test cases
2. Capture test output (stdout/stderr)
3. Document any failures with specific error messages
4. Save evidence to: `{missionDir}/evidence/core-algorithms/{group-id}/`

### Report Format

Write JSON report to: `.factory/validation/core-algorithms/user-testing/flows/{group-id}.json`

```json
{
  "groupId": "tpq-assertions",
  "assertionsTested": ["VAL-TPQ-001", "VAL-TPQ-002", ...],
  "results": {
    "VAL-TPQ-001": {
      "status": "pass|fail|blocked",
      "evidence": "test output or error message"
    }
  },
  "frictions": [],
  "blockers": [],
  "toolsUsed": ["pytest"]
}
```

## Assertion to Test Mapping

| Assertion | Test File | Test Class/Method |
|-----------|-----------|-------------------|
| VAL-TPQ-001 | test_tpq.py | TestPolarTransform |
| VAL-TPQ-002 | test_tpq.py | TestRecursiveCompression |
| VAL-TPQ-003 | test_tpq.py | TestBitAllocation |
| VAL-TPQ-004 | test_tpq.py | TestTemporalRedundancy |
| VAL-TPQ-005 | test_tpq.py | TestRoundtripAccuracy |
| VAL-SQJL-001 | test_sqjl.py | TestJLDistancePreservation |
| VAL-SQJL-002 | test_sqjl.py | TestSignBitQuantization |
| VAL-SQJL-003 | test_sqjl.py | TestUnbiasedEstimator |
| VAL-SQJL-004 | test_sqjl.py | TestSpatialRelationshipPreservation |
| VAL-MAMP-001 | test_mamp.py | TestLayerPrecisionAssignment |
| VAL-MAMP-002 | test_mamp.py | TestTimestepAwareAllocation |
| VAL-MAMP-004 | test_mamp.py | TestCrossAttentionHighPrecision |
| VAL-MAMP-005 | test_mamp.py | TestTemporalConsistencyOptimization |
| VAL-INT-001 | test_pipeline.py | TestPipelineExecution |
| VAL-CROSS-001 | test_pipeline.py | TestPipelineTensorFlow |
| VAL-INT-003 | test_kernels.py | CPU kernel performance tests |

## Running Tests

```bash
# Run specific test file
pytest tests/test_tpq.py -v

# Run specific test class
pytest tests/test_tpq.py::TestPolarTransform -v

# Run specific test method
pytest tests/test_tpq.py::TestPolarTransform::test_cartesian_to_polar_accuracy -v
```
