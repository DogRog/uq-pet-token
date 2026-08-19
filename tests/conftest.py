"""Shared offline fixtures."""

import os

import pytest
from typeguard import install_import_hook

install_import_hook("uq_pet")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture
def sample_examples() -> list[dict]:
    """Four PET-shaped sentences; two documents share sentence-ID 3."""
    return [
        {
            "document name": "doc-1",
            "sentence-ID": 3,
            "tokens": ["The", "clerk", "signs"],
            "ner-tags": [1, 2, 3],
        },
        {
            "document name": "doc-2",
            "sentence-ID": 3,
            "tokens": ["He", "sends", "it"],
            "ner-tags": [1, 3, 5],
        },
        {
            "document name": "doc-2",
            "sentence-ID": 4,
            "tokens": ["If", "valid"],
            "ner-tags": [9, 11],
        },
        {
            "document name": "doc-3",
            "sentence-ID": 0,
            "tokens": ["Done"],
            "ner-tags": [0],
        },
    ]
