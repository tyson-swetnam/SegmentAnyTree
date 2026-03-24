import pytest
from unittest.mock import patch

def test_profile_selection_large_gpu():
    from sat.gpu_profile import select_profile
    with patch("sat.gpu_profile._get_gpu_memory_gb", return_value=80.0):
        profile = select_profile()
        assert profile["cluster_nsample"] >= 200
        assert profile["num_workers"] >= 4

def test_profile_selection_small_gpu():
    from sat.gpu_profile import select_profile
    with patch("sat.gpu_profile._get_gpu_memory_gb", return_value=8.0):
        profile = select_profile()
        assert profile["cluster_nsample"] <= 64

def test_profile_selection_no_gpu():
    from sat.gpu_profile import select_profile
    with patch("sat.gpu_profile._get_gpu_memory_gb", return_value=0):
        profile = select_profile()
        assert profile["cluster_nsample"] == 32
