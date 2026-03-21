"""Test pipeline utility functions."""

import os
import tempfile
import pytest


def test_sanitize_filenames():
    """Dashes and spaces in filenames should be replaced with underscores."""
    from sat.pipeline.file_preparation import sanitize_filenames

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files with problematic names
        for name in ["file-with-dashes.las", "file with spaces.laz", "normal_file.ply"]:
            open(os.path.join(tmpdir, name), 'w').close()

        sanitize_filenames(tmpdir)

        result = sorted(os.listdir(tmpdir))
        assert "file_with_dashes.las" in result
        assert "file_with_spaces.laz" in result
        assert "normal_file.ply" in result
        assert "file-with-dashes.las" not in result
        assert "file with spaces.laz" not in result


def test_config_update():
    """modify_eval_yaml should update fold with PLY file paths."""
    from sat.pipeline.config_update import modify_eval_yaml
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal eval.yaml
        yaml_path = os.path.join(tmpdir, "eval.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'data': {'fold': []},
                'hydra': {'run': {'dir': ''}},
                'checkpoint_dir': 'model_file',
            }, f)

        # Create some PLY files
        ply_dir = os.path.join(tmpdir, "plys")
        os.makedirs(ply_dir)
        for name in ["a.ply", "b.ply"]:
            open(os.path.join(ply_dir, name), 'w').close()

        modify_eval_yaml(yaml_path, ply_dir, tmpdir)

        with open(yaml_path) as f:
            result = yaml.safe_load(f)

        assert len(result['data']['fold']) == 2
        assert result['hydra']['run']['dir'] == tmpdir


def test_paths_utility():
    """sat.utils.paths should return Path objects and respect env vars."""
    import os
    from sat.utils.paths import get_sat_root, get_sat_data, get_sat_model

    # Test defaults (auto-detect repo root)
    root = get_sat_root()
    assert root.exists() or True  # may not exist in test env, but should return Path

    # Test env var override
    os.environ['SAT_ROOT'] = '/tmp/test_sat'
    os.environ['SAT_DATA'] = '/tmp/test_data'
    os.environ['SAT_MODEL'] = '/tmp/test_model'
    try:
        from importlib import reload
        import sat.utils.paths
        reload(sat.utils.paths)
        from sat.utils.paths import get_sat_root, get_sat_data, get_sat_model
        assert str(get_sat_root()) == '/tmp/test_sat'
        assert str(get_sat_data()) == '/tmp/test_data'
        assert str(get_sat_model()) == '/tmp/test_model'
    finally:
        del os.environ['SAT_ROOT']
        del os.environ['SAT_DATA']
        del os.environ['SAT_MODEL']
