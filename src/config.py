from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_MAX_NEW_TOKENS: int = 1024
LLM_TEMPERATURE: float = 0.7
LLM_TOP_P: float = 0.9
LLM_REPETITION_PENALTY: float = 1.1
CHAT_MEMORY_TOKEN_LIMIT: int = 4896
#LLM_QUESTION: str = "What is the most famous tower in france?"
LLM_SYSTEM_PROMPT: str = """You are a factual research assistant specialized in Python programming.
 Answer using ONLY the provided source passages. 
 If the information is not in the passages,
 respond with: "I don't know."
 Be concise and precise. Structure every answer with: 1)
 Short direct answer (1–3 sentences); 2)
 Explanation with inline citations to source IDs and short quoted snippets; 3) 
 Up to 3 suggested follow-up questions. Always include a confidence level (High/Medium/Low) 
 and list full source metadata for every cited source."""

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 2
# The size of each text chunk in tokens
CHUNK_SIZE: int = 768
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 115

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"

# --- Reranker Configuration ---
RERANKER_TOP_N = 3  # For example, The number of nodes to return after reranking
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" # Or another cross-encoder model