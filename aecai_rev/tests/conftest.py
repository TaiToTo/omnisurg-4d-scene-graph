"""Shared fixtures: the configuration and the tagged evaluator.

These tests compare `aecai_rev` with the tagged AE-CAI code, so they need
the environment `aecai_rev/config.yaml` names; they raise without it.
"""
import pytest

from aecai_rev.config import load
from aecai_rev.tagged import use


@pytest.fixture(scope="session")
def cfg():
    return load()


@pytest.fixture(scope="session")
def et(cfg):
    return use(cfg)
