"""In-process (white-box) LLM sampling via transformers, for CUDA servers.

The Linux/GPU counterpart of mlx_scoring: local generation exposes the raw
logits at every step, so alongside each sampled text we record the per-token
predictive entropy (bits) that white-box uncertainty metrics need. Runs on
CUDA when available (falls back to MPS/CPU via train.get_device).

Batch-1 autoregressive decoding is memory-bandwidth-bound and leaves a GPU
almost idle, so every call generates `batch_size` sentences × num_samples
completions as one batched `generate`: prompts are prefilled together and all
rows decode in parallel.
"""

import math

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from .train import get_device


def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (bits) of the softmax distributions along the last axis."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return -(logprobs.exp() * logprobs).sum(dim=-1) / math.log(2)


def generated_length(tokens: torch.Tensor, stop_ids: list[int]) -> int:
    """Tokens up to and including the first stop token; anything after it is
    padding emitted while other rows of the batch were still generating."""
    stops = torch.isin(tokens, torch.tensor(stop_ids, device=tokens.device)).nonzero()
    return int(stops[0]) + 1 if stops.numel() else tokens.shape[0]


class RawEntropyTemperature(LogitsProcessor):
    """Records raw-logit entropy at every decode step, then applies temperature.

    One processor does both so the recorded entropy stays un-tempered
    (matching the mlx backend) without `output_logits=True`, which would pin
    every step's full-vocab logits in GPU memory for the whole generation —
    at useful batch widths that alone is tens of GB. This keeps one float per
    row per step, and never syncs the device mid-generation.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature
        self.step_entropies: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        self.step_entropies.append(entropy_bits(scores))
        return scores / self.temperature


class HFGenerator:
    """Loads a local HF model once and samples completions with per-token entropies."""

    def __init__(self, model_name: str, batch_size: int = 8, quantization: str | None = None):
        self.batch_size = batch_size
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Decoder-only batching: prompts of different lengths pad on the left
        # so every row's generation starts at the same position.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict = {"dtype": "auto"}
        if quantization is not None:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            # bitsandbytes places the weights at load time; .to() on a
            # quantized model raises.
            kwargs["device_map"] = str(self.device)
        self.model = self._load_model(model_name, **kwargs)
        if quantization is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_model(model_name: str, **kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except ValueError:
            # Multimodal checkpoints (e.g. gemma image-text-to-text) register
            # only with the multimodal auto class; text-only generate on them
            # works the same as on a causal LM.
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

    def _stop_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return sorted({*eos, self.tokenizer.pad_token_id})

    def _chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            # Qwen3 defaults to thinking mode; <think> blocks would break NER
            # parsing and pollute the entropy signal with reasoning tokens.
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

    @torch.no_grad()
    def sample_batch(
        self, prompts: list[str], temperature: float, max_tokens: int, seeds: list[list[int]]
    ) -> list[tuple[list[str], list[list[float]]]]:
        """One batched generate call: len(seeds[i]) sampled completions of each
        prompt; returns per-prompt (texts, per-generated-token entropy lists).

        All rows of a batched generate draw from one shared RNG stream, so
        per-sample seeding is impossible: the batch is seeded once with
        seeds[0][0]. Draws therefore depend on how sentences are batched
        together, but batching is deterministic (see _score_pending_local),
        so an interrupted and resumed pass still reproduces the same draws.
        """
        num_samples = len(seeds[0])
        text_prompts = [self._chat_prompt(p) for p in prompts]
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"].repeat_interleave(num_samples, dim=0)
        attention_mask = inputs["attention_mask"].repeat_interleave(num_samples, dim=0)

        recorder = RawEntropyTemperature(temperature)
        torch.manual_seed(seeds[0][0])
        sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            # Neutralize the model's shipped sampler defaults (Qwen3 sets
            # temperature=0.6, top_k=20, top_p=0.95): the recorder applies our
            # temperature itself after measuring raw entropy, giving pure
            # temperature sampling like the mlx and API backends.
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            logits_processor=LogitsProcessorList([recorder]),
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = sequences[:, input_ids.shape[1] :].cpu()
        # [rows, steps] in one device-to-host transfer.
        step_entropies = torch.stack(recorder.step_entropies, dim=1).cpu()

        stop_ids = self._stop_ids()
        results = []
        for p in range(len(prompts)):
            texts, entropies = [], []
            for row in range(p * num_samples, (p + 1) * num_samples):
                length = generated_length(generated[row], stop_ids)
                text = self.tokenizer.decode(generated[row][:length], skip_special_tokens=True)
                texts.append(text.strip())
                entropies.append([round(float(e), 4) for e in step_entropies[row][:length]])
            results.append((texts, entropies))
        return results
