from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vdb._src.common import MissingDependency
from vdb._src.types import Embedder

try:
    import openai  # type: ignore
except ModuleNotFoundError:
    openai = MissingDependency("OpenAI", "openai")  # type: ignore
try:
    import tiktoken  # type: ignore
except ModuleNotFoundError:
    tiktoken = MissingDependency("TikToken", "tiktoken")  # type: ignore

_DEFAULT_MODEL_NAME = "text-embedding-ada-002"
_DEFAULT_TOKENIZER_NAME = "cl100k_base"
_DEFAULT_MAX_TOKENS = 8191


class OpenAIEmbedder(Embedder[str]):
    def __init__(
        self,
        api_key: str,
        model_name: str = _DEFAULT_MODEL_NAME,
        tokenizer_name: Optional[str] = _DEFAULT_TOKENIZER_NAME,
        max_tokens: Optional[int] = _DEFAULT_MAX_TOKENS,
        create_fn: Callable[..., Dict[str, Any]] = openai.Embedding.create,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.tokenizer = (
            None if tokenizer_name is None else tiktoken.get_encoding(tokenizer_name)
        )
        if max_tokens is not None:
            if tokenizer_name is None:
                raise ValueError("Cannot specify max_tokens without tokenizer_name")
        self.max_tokens = max_tokens
        self.create_fn = create_fn

    def __call__(self, items: Sequence[str]) -> np.ndarray:
        """Embeds a sequence of strings.

        Args:
            items: A sequence of strings to embed.

        Returns:
            An array of shape (len(items), d) containing the embeddings. Here,
            d depends on the model chosen.
        """
        items = self._validate_items(items)
        responses = self.create_fn(
            inputs=items,
            api_key=self.api_key,
            model=self.model_name,
        )
        embeddings_list = [data["embedding"] for data in responses["data"]]
        return np.asarray(embeddings_list, dtype=np.float32)

    def tokenize(self, items: Sequence[str]) -> List[np.ndarray]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not specified")
        tokens = self.tokenizer.encode_batch(list(items))
        return [np.asarray(token, dtype=np.int32) for token in tokens]

    def _validate_items(self, items: Sequence[str]) -> List[str]:
        items = list(items)
        if self.max_tokens is None:
            return items
        tokens_batch = self.tokenize(items)
        lengths = [len(tokens) for tokens in tokens_batch]
        if any(length > self.max_tokens for length in lengths):
            raise ValueError(
                f"Expected all items to contain at most "
                f"{self.max_tokens} tokens, but got {lengths}"
            )
        return items