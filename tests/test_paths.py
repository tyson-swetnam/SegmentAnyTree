"""Test that no hardcoded paths remain in the codebase.

Scans all .py, .sh, .yaml files for paths that only work on the
original author's machine.
"""

import os
import re
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

FORBIDDEN_PATTERNS = [
    r'/home/nibio/',
    r'/home/datascience',
    r'/home/nibio/mutable-outside-world',
    r'PanopticSegForLargeScalePointCloud',
]

# Files that are allowed to contain forbidden patterns (documentation, plan, old files)
EXCLUDED_FILES = {
    'PLAN.md',
    'CLAUDE.md',
    '.hydra/overrides.yaml',  # checkpoint metadata
    '.hydra/config.yaml',
    'test_paths.py',  # this file contains the patterns as test data
}

# Directories to skip entirely
EXCLUDED_DIRS = {
    '.git', '__pycache__', 'model_file', '.ipynb_checkpoints',
    'outputs', 'wandb', 'data',
    # Old directories (kept for reference but not part of new pipeline)
    'nibio_inference', 'nibio_sparsify', 'bash_helpers',
    'big_table_creation', 'visualization', 'scripts/oracle',
}

# Old root-level files scheduled for deletion (not part of new pipeline)
EXCLUDED_FILES_ROOT = {
    'run_inference.sh', 'run_batch_inference.sh', 'run_oracle_pipeline.sh',
    'run_docker_locally.sh', 'run_paper_test.sh', 'compute_capacity.sh',
    'merge_all.sh', 'run_podman_with_gpu.sh', 'run_bash_in_podman_with_gpu.sh',
    'run_pipeline.sh', 'build.sh', 'oracle_wrapper.py',
    'evaluation_stats_FOR.py', 'evaluation_stats_NPM3D.py',
    'sample_data_conversion.py',
}

SCAN_EXTENSIONS = {'.py', '.sh', '.yaml', '.yml'}


def _collect_files():
    """Collect all scannable files in the repo."""
    files = []
    for root, dirs, filenames in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fname in filenames:
            if any(fname.endswith(ext) for ext in SCAN_EXTENSIONS):
                rel = os.path.relpath(os.path.join(root, fname), REPO_ROOT)
                if not any(excl in rel for excl in EXCLUDED_FILES):
                    # Skip old root-level files scheduled for deletion
                    if os.path.dirname(os.path.join(root, fname)) == str(REPO_ROOT) and fname in EXCLUDED_FILES_ROOT:
                        continue
                    # Skip scripts/oracle/
                    if 'scripts/oracle' in rel:
                        continue
                    files.append(os.path.join(root, fname))
    return files


@pytest.mark.parametrize("filepath", _collect_files(), ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_no_hardcoded_paths(filepath):
    """Each file should not contain hardcoded paths from the original author's machine."""
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()

    violations = []
    for i, line in enumerate(content.splitlines(), 1):
        # Skip comments in YAML and shell scripts
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                violations.append(f"  Line {i}: {line.strip()}")

    assert not violations, (
        f"Hardcoded paths found in {os.path.relpath(filepath, REPO_ROOT)}:\n"
        + "\n".join(violations)
    )
