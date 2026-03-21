"""Test point cloud I/O roundtrips."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Create a small synthetic point cloud DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'x': np.random.uniform(0, 10, n).astype(np.float32),
        'y': np.random.uniform(0, 10, n).astype(np.float32),
        'z': np.random.uniform(0, 5, n).astype(np.float32),
        'intensity': np.random.uniform(0, 1000, n).astype(np.float32),
    })


def test_ply_roundtrip(sample_dataframe):
    """PLY write then read should preserve data."""
    from sat.io.ply_io import pandas_to_ply, ply_to_pandas

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.ply")
        pandas_to_ply(sample_dataframe, path)
        result = ply_to_pandas(path)

    assert set(result.columns) == set(sample_dataframe.columns)
    assert len(result) == len(sample_dataframe)
    for col in sample_dataframe.columns:
        np.testing.assert_allclose(
            result[col].astype(float).values,
            sample_dataframe[col].values,
            atol=1e-3,
            err_msg=f"Column {col} mismatch in PLY roundtrip"
        )


def test_ply_to_pandas_missing_file():
    """Reading a nonexistent PLY file should raise."""
    from sat.io.ply_io import ply_to_pandas
    with pytest.raises(Exception):
        ply_to_pandas("/nonexistent/file.ply")
