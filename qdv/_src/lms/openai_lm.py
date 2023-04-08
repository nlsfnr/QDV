from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union

import numpy as np

from qdv._src.common import get_openai_key, try_import
from qdv._src.types import LanguageModel

if TYPE_CHECKING:
    import openai
    import tiktoken
else:
    openai = try_import("openai", "OpenAI", "openai")
    tiktoken = try_import("tiktoken", "TikToken", "tiktoken")

_DEFAULT_MODEL_NAME = "text-davinci-003"
_DEFAULT_MAX_PROMPT_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.0


class OpenAILanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: Union[None, str, Path] = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        max_prompt_tokens: Optional[int] = _DEFAULT_MAX_PROMPT_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        create_fn: Callable[..., Dict[str, Any]] = openai.Completion.create,
    ) -> None:
        self.api_key = get_openai_key(api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = temperature
        self.create_fn = create_fn

    def __call__(
        self,
        prompt: str,
        max_tokens: int,
    ) -> Iterable[str]:
        """Generate text from the given prompt."""
        prompt = self._validate_prompt(prompt)
        response = self.create_fn(
            prompt=prompt,
            max_tokens=max_tokens,
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )
        generation = response["choices"][0]["text"]
        yield generation

    def tokenize(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode(text)
        return np.asarray(tokens, dtype=np.int32)

    def _validate_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if self.max_prompt_tokens is None:
            return prompt
        tokens = self.tokenize(prompt)
        if len(tokens) > self.max_prompt_tokens:
            raise ValueError(
                f"Expected prompt to contain at most "
                f"{self.max_prompt_tokens} tokens, but got {len(tokens)}"
            )
        return prompt
