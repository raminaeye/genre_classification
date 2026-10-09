"""Shared pytest fixtures for the check_data step.

Downloads the reference and sample datasets as W&B artifacts once per
session and exposes them, plus the KS-test significance threshold, to the
tests in test_data.py.
"""
import pytest
import pandas as pd
import wandb


run = wandb.init(job_type="data_tests")


def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")


@pytest.fixture(scope="session")
def data(request):
    """(reference_df, sample_df) downloaded from W&B, once per session."""

    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    """KS-test significance threshold from the --ks_alpha CLI option."""
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)
