import pytest

from uq_pet.experiment import require_wandb_credentials


def test_online_wandb_launch_requires_api_key():
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({})
    with pytest.raises(ValueError, match="WANDB_API_KEY is missing"):
        require_wandb_credentials({"WANDB_API_KEY": "   "})


def test_wandb_preflight_accepts_key_or_offline_mode():
    require_wandb_credentials({"WANDB_API_KEY": "configured"})
    require_wandb_credentials({"WANDB_MODE": "offline"})
