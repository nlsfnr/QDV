from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from qdv._src.common import get_openai_key, try_import
from qdv._src.types import Embedder

if TYPE_CHECKING:
    import openai
    import tiktoken
else:
    openai = try_import("openai", "OpenAI", "openai")
    tiktoken = try_import("tiktoken", "TikToken", "tiktoken")

_DEFAULT_MODEL_NAME = "text-embedding-ada-002"
_DEFAULT_EMBEDDING_DIM = 1536
_DEFAULT_MAX_TOKENS = 8191


class OpenAIEmbedder(Embedder[str]):
    def __init__(
        self,
        api_key: Union[None, str, Path] = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        max_tokens: Optional[int] = _DEFAULT_MAX_TOKENS,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        create_fn: Callable[..., Dict[str, Any]] = openai.Embedding.create,
    ) -> None:
        self.api_key = get_openai_key(api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.create_fn = create_fn
        self._embedding_dim = embedding_dim

    @property
    def dim(self) -> int:
        return self._embedding_dim

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
            input=items,
            api_key=self.api_key,
            model=self.model_name,
        )
        embeddings_list = [data["embedding"] for data in responses["data"]]
        embeddings = np.asarray(embeddings_list, dtype=np.float32)
        return embeddings

    def tokenize(self, items: Sequence[str]) -> List[np.ndarray]:
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
