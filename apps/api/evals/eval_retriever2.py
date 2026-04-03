import asyncio
import math
import os
import time

from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult

from api.agents.retrieval_generation import rag_pipeline

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    IDBasedContextPrecision,
    IDBasedContextRecall,
    ResponseRelevancy,
)

# Environment variables
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Pause before each target (RAG) run to reduce rate-limit bursts from HF / Groq / Gemini
RAG_PIPELINE_DELAY_SECONDS = float(os.getenv("RAG_PIPELINE_DELAY_SECONDS", "30"))

ls_client = Client()
qdrant_client = QdrantClient(url=QDRANT_URL)

model = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=model,
    huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
)

groq_api_key_2 = os.getenv("GROQ_API_KEY2")

if groq_api_key_2:
    # Groq judge model for RAGAS (no langchain-groq dependency needed).
    from groq import Groq
    from langchain_core.callbacks import Callbacks
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import PromptValue

    class GroqRagasLLM(BaseRagasLLM):
        """Ragas LLM adapter backed by the Groq SDK."""

        def __init__(
            self,
            *,
            api_key: str,
            model: str = "qwen/qwen3-32b",
            run_config=None,
            cache=None,
        ):
            # Ragas expects a real RunConfig; passing None crashes in tenacity retry setup.
            from ragas.run_config import RunConfig

            super().__init__(run_config=run_config or RunConfig(), cache=cache)
            self._client = Groq(api_key=api_key)
            self._model = model

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _prompt_to_text(self, prompt: PromptValue) -> str:
            return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.01,
            stop=None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            prompt_text = self._prompt_to_text(prompt)
            generations: list[Generation] = []
            for _ in range(n):
                # RAGAS expects strict JSON for many prompts (pydantic parsing).
                # Use a strong system instruction and (when supported) request JSON-only output.
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON that matches the requested schema. "
                            "Do not add explanations, markdown, or extra text."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ]

                try:
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                        stop=stop,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    # Some Groq models/endpoints may not support response_format; fall back.
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                        stop=stop,
                    )
                text = completion.choices[0].message.content if completion.choices else ""
                generations.append(Generation(text=text or ""))

            # Ragas pydantic_prompt expects: generations[0][i].text for BaseRagasLLM
            return LLMResult(generations=[generations])

        async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.01,
            stop=None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )

    ragas_llm = GroqRagasLLM(
        api_key=groq_api_key_2,
        # Default to a generally JSON-compliant chat model; override via env if needed.
        model=os.getenv("GROQ_EVAL_LLM_MODEL2", "llama-3.1-8b-instant"),
    )
else:
    _gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not _gemini_key:
        raise ValueError(
            "Set GROQ_API_KEY2 (preferred) or GOOGLE_API_KEY/GEMINI_API_KEY in your .env to use a judge LLM for RAGAS."
        )

    # Force n=1 to avoid Gemini "Multiple candidates is not enabled for this model".
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=_gemini_key,
            n=1,
        )
    )

ragas_embeddings = LangchainEmbeddingsWrapper(hf)


def _is_rag_output_dict(d: object) -> bool:
    return (
        isinstance(d, dict)
        and "question" in d
        and "answer" in d
        and "retrieved_context" in d
    )


def _target_outputs(run) -> dict:
    """Resolve rag_pipeline dict from Run/RunTree (unwrap LangSmith output nesting + child runs)."""

    def from_outputs_block(out: dict) -> dict:
        if _is_rag_output_dict(out):
            return out
        for k in ("output", "result"):
            inner = out.get(k)
            if _is_rag_output_dict(inner):
                return inner
        return {}

    visited: set[int] = set()

    def walk(node) -> dict:
        if node is None:
            return {}
        nid = id(node)
        if nid in visited:
            return {}
        visited.add(nid)

        out = getattr(node, "outputs", None)
        if isinstance(out, dict):
            got = from_outputs_block(out)
            if got:
                return got
        for child in getattr(node, "child_runs", None) or ():
            got = walk(child)
            if got:
                return got
        return {}

    return walk(run)


def _example_fields(example) -> dict:
    """Merge Example.inputs and Example.outputs so labels are found regardless of storage."""
    if example is None:
        return {}
    merged: dict = {}
    for attr in ("inputs", "outputs"):
        block = getattr(example, attr, None)
        if isinstance(block, dict):
            merged.update(block)
    return merged


def _reference_context_ids(fields: dict) -> list:
    for key in ("reference_context_ids", "chunk_ids", "relevant_context_ids"):
        v = fields.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    return []


def _eval_score(key: str, raw) -> EvaluationResult:
    """Coerce Ragas / NumPy scores to JSON-safe floats; explicit keys help LangSmith experiment columns."""
    if raw is None:
        return EvaluationResult(key=key, score=None, comment="no score")
    try:
        val = raw.item() if hasattr(raw, "item") and callable(raw.item) else raw
        s = float(val)
    except (TypeError, ValueError):
        return EvaluationResult(key=key, score=None, comment=f"non-numeric score: {raw!r}")
    if math.isnan(s):
        return EvaluationResult(key=key, score=None, comment="skipped (NaN or missing reference IDs)")
    return EvaluationResult(key=key, score=s)


def run_rag_with_rate_limit_spacing(inputs: dict):
    """Sleep before each traced RAG call so upstream APIs see lower QPS."""
    if RAG_PIPELINE_DELAY_SECONDS > 0:
        time.sleep(RAG_PIPELINE_DELAY_SECONDS)
    return rag_pipeline(inputs["question"], qdrant_client)


def ragas_faithfulness(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_faithfulness",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        sample = SingleTurnSample(
            user_input=o["question"],
            response=o["answer"],
            retrieved_contexts=o["retrieved_context"],
        )
        scorer = Faithfulness(llm=ragas_llm)
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_faithfulness", asyncio.run(_score()))


def ragas_response_relevancy(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_response_relevancy",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        sample = SingleTurnSample(
            user_input=o["question"],
            response=o["answer"],
            retrieved_contexts=o["retrieved_context"],
        )
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_response_relevancy", asyncio.run(_score()))


def ragas_context_precision_id_based(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_context_precision_id_based",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            retrieved_context_ids=o["retrieved_context_ids"],
            reference_context_ids=ref_ids,
        )
        scorer = IDBasedContextPrecision()
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_context_precision_id_based", asyncio.run(_score()))


def ragas_context_recall_id_based(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_context_recall_id_based",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            retrieved_context_ids=o["retrieved_context_ids"],
            reference_context_ids=ref_ids,
        )
        scorer = IDBasedContextRecall()
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_context_recall_id_based", asyncio.run(_score()))


results = ls_client.evaluate(
    run_rag_with_rate_limit_spacing,
    data="rag-evaluation-dataset",
    evaluators={
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
    },
    experiment_prefix="retriever",
    max_concurrency=1,
)
