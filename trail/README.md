# Trail Directory

This directory contains obsolete, superseded, or non-working code that is kept for reference but should not be used.

## Contents

### run_maml_meta_learning.py (v1)
**Status**: Superseded by v2 in `future/`

First version of MAML implementation. Has issues with parameter copying efficiency. The v2 version in `future/` uses the `higher` library for better performance.

### run_pino_2d.py
**Status**: Superseded by full version in `future/`

Simplified 2D PINO implementation that doesn't properly handle spatial grid mapping. The full version in `future/` has complete RBF interpolation and FNO architecture.

### pino2d_full_output.txt
**Status**: Temporary output file

Training log from PINO-2D experiments. Kept for debugging reference.

## Why Keep These?

1. **Debugging reference**: If similar issues arise in new implementations
2. **Development history**: Shows evolution of the codebase
3. **Learning resource**: For understanding what didn't work

## Do Not Use

Code in this directory is **not maintained** and may:
- Have bugs
- Use outdated APIs
- Produce incorrect results
- Be incompatible with current data format

Always use the versions in the repository root or `future/` directory.
