from pathlib import Path
import time

from ragas.metrics.base import Metric
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,

)

# --- LLM Model Configuration ---
EVALUATION_LLM_MODEL: str = "moonshotai/kimi-k2-instruct"

# --- Embedding Model Configuration ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- Paths for Evaluation ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results/"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_vector_stores/"
)
EVALUATION_EMBEDDING_CACHE_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_embedding_models/"
)

# --- Ragas Evaluation Metrics ---
EVALUATION_METRICS: list[Metric] = [
    Faithfulness(),
    AnswerCorrectness(),
    ContextPrecision(),
    ContextRecall(),
]

# --- Sleep Timers for API Limits ---
SLEEP_PER_EVALUATION: int = 120
SLEEP_PER_QUESTION: int = 6
MAX_RETRIES = 4
BACKOFF_FACTOR = 2  # multiplies wait each retry


def sleep_with_backoff(base_sleep: int):
    for attempt in range(MAX_RETRIES):
        wait = min(base_sleep * (BACKOFF_FACTOR ** attempt), SLEEP_PER_EVALUATION)
        time.sleep(wait)
        # after waking, try the operation and break on success
        try:
            return True  # replace with actual operation invocation
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


# --- Configuration for Chunking Strategy Evaluation ---
CHUNKING_STRATEGY_CONFIGS: list[dict[str, int]] = [
    {'size': 768, 'overlap': 115},
    {'size': 768, 'overlap': 115},
    {'size': 1024, 'overlap': 200},
    {'size': 896, 'overlap': 128},
]


# --- Cross-encoder Model for Reranking ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --- Configuration for Reranking Evaluation ---
RERANKER_CONFIGS: list[dict[str, int]] = [
    {'retriever_k': 10, 'reranker_n': 2},
    {'retriever_k': 10, 'reranker_n': 5},
    {'retriever_k': 20, 'reranker_n': 5},
]


# --- Query Rewrite Evaluation ---
# The 'best' reranker strategy found in the previous evaluation stage.
# IMPORTANT: You must update this with the values you found to be optimal.
BEST_RERANKER_STRATEGY: dict[str, int] = {'retriever_k': 20, 'reranker_n': 5}