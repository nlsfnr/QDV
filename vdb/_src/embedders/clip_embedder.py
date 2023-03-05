from typing import Sequence

import numpy as np

from vdb._src.common import MissingDependency
from vdb._src.types import ArrayLike, Embedder

try:
    import transformers  # type: ignore

    transformers.logging.set_verbosity_error()
except ModuleNotFoundError:
    transformers = MissingDependency("Transformers", "transformers")  # type: ignore
try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = MissingDependency("PyTorch", "torch")  # type: ignore


_DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
_DEFAULT_TOKENIZER_NAME = _DEFAULT_MODEL_NAME
_DEFAULT_EMBEDDING_DIM = 512
_DEFAULT_MAX_TOKENS = 77


class CLIPTextEmbedder(Embedder[str]):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        tokenizer_name: str = _DEFAULT_TOKENIZER_NAME,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = transformers.CLIPTextModelWithProjection.from_pretrained(
            model_name
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.device = device
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
        inputs = self._validate_and_process_items(items)
        with torch.no_grad(), torch.autocast(self.device):
            outputs = self.model(**inputs)
        return outputs.text_embeds.detach().numpy().astype(np.float32)

    def _validate_and_process_items(
        self, items: Sequence[str]
    ) -> transformers.BatchEncoding:
        items = list(items)
        encodings = self.tokenizer(items, return_tensors="pt", padding=True)
        lengths = [len(encoding) for encoding in encodings.input_ids]
        if any(length > self.max_tokens for length in lengths):
            raise ValueError(
                f"Expected all items to contain at most "
                f"{self.max_tokens} tokens, but got {lengths}"
            )
        return encodings


class CLIPImageEmbedder(Embedder[ArrayLike]):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        processor_name: str = _DEFAULT_MODEL_NAME,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = transformers.CLIPVisionModelWithProjection.from_pretrained(
            model_name
        )
        self.processor = transformers.AutoProcessor.from_pretrained(processor_name)
        self.device = device
        self._embedding_dim = embedding_dim

    @property
    def dim(self) -> int:
        return self._embedding_dim

    def __call__(self, items: ArrayLike) -> np.ndarray:
        """Embeds a sequence of images.

        Args:
            items: A sequence of images to embed. Each image should be a
                numpy array of shape (H, W, 3), where H and W are the height
                and width of the image, respectively.

        Returns:
            An array of shape (len(items), d) containing the embeddings. Here,
            d depends on the model chosen.
        """
        inputs = self._validate_and_process_items(items)
        with torch.no_grad(), torch.autocast(self.device):
            outputs = self.model(**inputs)
        return outputs.image_embeds.detach().numpy().astype(np.float32)

    def _validate_and_process_items(
        self, items: ArrayLike
    ) -> transformers.BatchEncoding:
        items = [np.asarray(item) for item in items]
        shapes = [item.shape for item in items]
        if any(len(shape) != 3 for shape in shapes):
            raise ValueError(
                f"Expected all items to be 3-dimensional, but got shapes {shapes}"
            )
        inputs = self.processor(images=items, return_tensors="pt")
        return inputs
