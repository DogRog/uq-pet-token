"""Fixtures shared across test modules."""

import os
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from uq_pet.config import ExperimentConfig
from uq_pet.data_prep import NER_TAGS
from uq_pet.token_model import set_seed


@pytest.fixture
def tiny_experiment(tmp_path):
    """Exercise real model loading and spawned training without network access."""
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "b": 5}))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(tmp_path)
    set_seed(72)
    model = BertForTokenClassification(
        BertConfig(
            vocab_size=6,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            num_labels=len(NER_TAGS),
            pad_token_id=3,
        )
    )
    model.save_pretrained(tmp_path)
    examples = [
        {"tokens": ["a", "b"], "ner_tags": [1, 0]},
        {"tokens": ["b", "a", "a"], "ner_tags": [0, 1, 2]},
    ]
    pool = [
        {"pool_idx": idx, "document_name": "tiny", "sentence_id": idx, "tokens": words}
        for idx, words in enumerate((["a", "b"], ["b", "a", "b"], ["a", "a"]))
    ]
    gold = {
        (idx, word): word % 3
        for idx, item in enumerate(pool)
        for word in range(len(item["tokens"]))
    }
    config = ExperimentConfig(
        checkpoint=str(tmp_path),
        model_seeds=[11, 3, 8],
        k=2,
        bootstrap_epochs=1,
        batch_size=2,
        score_batch_size=2,
        max_length=8,
    )
    return (examples, pool, gold, examples), {
        **config.active_learning_kwargs(),
        "device": torch.device("cpu"),
    }


@pytest.fixture
def fake_wandb(monkeypatch):
    class Run:
        def __init__(self, settings):
            self.settings = settings
            self.logs = []
            self.metrics = []
            self.exit_codes = []

        def log(self, payload):
            self.logs.append(payload)

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def finish(self, *, exit_code):
            self.exit_codes.append(exit_code)

        @property
        def project_url(self):
            return "https://wandb.ai/test/pet"

        @property
        def url(self):
            return f"https://wandb.ai/test/pet/runs/{self.settings['name']}"

    class SDK:
        runs = []
        logins = []

        def init(self, **settings):
            run = Run(settings)
            self.runs.append(run)
            return run

        sweeps = []

        def sweep(self, definition, **kwargs):
            # Like wandb.sweep(): print a banner and export the target for later inits.
            print("Create sweep with ID: sweep")
            os.environ["WANDB_ENTITY"] = kwargs["entity"]
            os.environ["WANDB_PROJECT"] = kwargs["project"]
            self.sweeps.append((definition, kwargs))
            return f"sweep-{len(self.sweeps)}"

        setups = []

        def Settings(self, **settings):
            return settings

        def setup(self, settings=None):
            self.setups.append(settings)

        deleted_sweeps = set()
        warnings = []
        tag_updates = []

        def Api(self):
            sdk = self

            class SavedRun:
                def __init__(self, path):
                    self.path = path
                    self.tags = []

                def update(self):
                    sdk.tag_updates.append((self.path, list(self.tags)))

            class API:
                default_entity = "test"

                def sweep(self, path):
                    if path.rsplit("/", 1)[-1] in sdk.deleted_sweeps:
                        raise ValueError(f"Could not find sweep <Sweep {path}>")

                def run(self, path):
                    return SavedRun(path)

            return API()

        def termwarn(self, message):
            self.warnings.append(message)

        def login(self, **kwargs):
            self.logins.append(kwargs)

    from uq_pet.utils import wandb_tuning

    class Workspace:
        url = "https://wandb.ai/test/pet/workspace?view=tuning"

        def save(self):
            pass

    monkeypatch.setattr(wandb_tuning, "chart_workspace", lambda *args: Workspace())
    monkeypatch.setattr(wandb_tuning, "supervised_chart_workspace", lambda *args: Workspace())
    sdk = SDK()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    for name in ("WANDB_MODE", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    return sdk
