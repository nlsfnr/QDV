from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple, Union
from dataclasses import dataclass

from qdv._src.common import get_openai_key
from qdv._src.embedders import OpenAIEmbedder
from qdv._src.indices import KNNIndex
from qdv._src.item_stores import JsonStore
from qdv._src.lms import OpenAILanguageModel
from qdv._src.types import Embedder, Index, LanguageModel


_DEFAULT_QA_TASK_PROMPT = "Truthful question answering based on top-k documents."


@dataclass
class QASolutionResult:
    question: str
    answer: str
    ids: Tuple[str, ...]
    texts: Tuple[str, ...]
    distances: Tuple[float, ...]
    task_prompt: str
    prompt: str


class QASolution:
    """Question answering on a set of documents."""

    def __init__(
        self,
        language_model: LanguageModel,
        embedder: Embedder,
        index: Index,
        ids_to_texts: Callable[[Sequence[str]], Sequence[str]],
    ) -> None:
        self.language_model = language_model
        self.embedder = embedder
        self.index = index
        self.ids_to_texts = ids_to_texts

    def __call__(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 256,
        task_prompt: str = _DEFAULT_QA_TASK_PROMPT,
    ) -> QASolutionResult:
        """Answer a question, using the top-k results in the index as context."""
        embeddings = self.embedder([question])
        (ids,), (distances,) = self.index.query(embeddings, top_k)
        texts = self.ids_to_texts(ids)
        prompt = self.prepare_prompt(texts, question, task_prompt)
        answer = self.language_model.call_and_collect(prompt, max_tokens=max_tokens).strip()
        return QASolutionResult(
            question=question,
            answer=answer,
            ids=tuple(ids),
            texts=tuple(texts),
            distances=tuple(distances),
            task_prompt=task_prompt,
            prompt=prompt,
        )

    def prepare_prompt(
        self, texts: Iterable[str], question: str, task_prompt: str
    ) -> str:
        texts = [
            f"[[ DOCUMENT {i + 1} ]]\n\n{t.strip()}\n\n" for i, t in enumerate(texts)
        ]
        joint_texts = "".join(texts)
        return (
            f"[[ TASK ]]\n\n{task_prompt.strip()}\n\n"
            f"{joint_texts}"
            f"[[ QUESTION ]]\n\n{question.strip()}\n\n"
            "[[ ANSWER ]]\n\n"
        )


@dataclass
class OpenAIE2ESolutionResult(QASolutionResult):
    pass


class OpenAIE2ESolution:
    def __init__(
        self,
        language_model: LanguageModel,
        embedder: Embedder,
        index: KNNIndex,
        text_store: JsonStore[str],
    ) -> None:
        self.language_model = language_model
        self.embedder = embedder
        self.index = index
        self.text_store = text_store

    @classmethod
    def from_item_store(
        cls,
        text_store: JsonStore[str],
        api_key: Union[None, str, Path] = None,
    ) -> OpenAIE2ESolution:
        api_key = get_openai_key(api_key)
        embedder = OpenAIEmbedder(api_key)
        ids = list(text_store.ids)
        embeddings = embedder(text_store.retrieve(ids))
        index = KNNIndex.from_data(ids, embeddings)
        lm = OpenAILanguageModel(api_key)
        return cls(
            language_model=lm, embedder=embedder, index=index, text_store=text_store
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path / "index")
        self.text_store.save(path / "text_store.json")

    @classmethod
    def load(
        cls,
        path: Path,
        api_key: Union[None, str, Path] = None,
    ) -> OpenAIE2ESolution:
        api_key = get_openai_key(api_key)
        index = KNNIndex.load(path / "index")
        embedder = OpenAIEmbedder(api_key)
        text_store = JsonStore[str].load(path / "text_store.json")
        lm = OpenAILanguageModel(api_key)
        return cls(
            language_model=lm, embedder=embedder, index=index, text_store=text_store
        )

    def __call__(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 256,
        task_prompt: str = _DEFAULT_QA_TASK_PROMPT,
    ) -> OpenAIE2ESolutionResult:
        qa = QASolution(
            language_model=self.language_model,
            embedder=self.embedder,
            index=self.index,
            ids_to_texts=self.text_store.retrieve,
        )
        answer = qa(question, top_k, max_tokens, task_prompt)
        return OpenAIE2ESolutionResult(
            question=answer.question,
            answer=answer.answer,
            ids=answer.ids,
            texts=answer.texts,
            distances=answer.distances,
            task_prompt=answer.task_prompt,
            prompt=answer.prompt,
        )
