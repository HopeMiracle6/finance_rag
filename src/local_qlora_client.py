from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.logging_utils import logger
from src.utils import resolve_path


class LocalQLoRAClient:
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-4B",
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = resolve_path(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.tokenizer: Any = None
        self.model: Any = None
        self._load()

    def _load(self) -> None:
        if self.adapter_path and not self.adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter 不存在: {self.adapter_path}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        if self.load_in_4bit:
            if not torch.cuda.is_available():
                logger.warning("当前 torch 不支持 CUDA，已跳过 4-bit bitsandbytes 加载")
            else:
                from transformers import BitsAndBytesConfig

                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

        local_files_only = os.getenv("FINANCE_RAG_ALLOW_MODEL_DOWNLOAD", "0") != "1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=self.trust_remote_code,
            local_files_only=local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
            "local_files_only": local_files_only,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        base = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path)) if self.adapter_path else base
        self.model.eval()
        logger.info(f"已加载本地 QLoRA: base={self.base_model}, adapter={self.adapter_path}")

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LocalQLoRAClient 尚未加载模型")

        inputs = self._apply_chat_template(messages)
        device = self._input_device()
        if isinstance(inputs, Mapping):
            model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = inputs.to(device) if hasattr(inputs, "to") else inputs
            model_inputs = {"input_ids": input_ids}
        if "attention_mask" not in model_inputs and hasattr(input_ids, "new_ones"):
            model_inputs["attention_mask"] = input_ids.new_ones(input_ids.shape)

        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**model_inputs, **generation_kwargs)
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> Any:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _input_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device
