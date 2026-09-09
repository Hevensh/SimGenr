# Validation evidence

These compact artifacts accompany [the validation report](../VALIDATION_RESULTS.md).
The three scenario reports and plot are copied from the final local runs without
changing numerical results. The pytest transcript uses a repository-relative path
and the JUnit copy omits the machine hostname; all test cases, counts and timings
are preserved.

Large generated arrays, installed dependencies and temporary files remain excluded
from Git under `outputs/`. Run `scripts/run_physics_checks.ps1` to reproduce tests,
and follow [the generation instructions](../PHYSICS_REVIEW_ZH.md#复现) to regenerate
the worlds and validate their constraints.
