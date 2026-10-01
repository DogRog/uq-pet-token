import pytest

from uq_pet.utils.wandb_logging import require_wandb_credentials


def test_online_wandb_launch_requires_api_key():
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({})
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({"WANDB_API_KEY": "   "})


def test_wandb_preflight_accepts_key_or_offline_mode():
    require_wandb_credentials({"WANDB_API_KEY": "configured"})
    require_wandb_credentials({"WANDB_MODE": "offline"})


def test_installed_wandb_run_exposes_url_properties():
    """The test fake mirrors these attributes; wandb 0.30 removed get_url()."""
    from wandb.sdk.wandb_run import Run

    assert isinstance(Run.url, property)
    assert isinstance(Run.project_url, property)
