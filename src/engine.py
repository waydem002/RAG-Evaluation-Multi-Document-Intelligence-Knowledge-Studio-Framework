from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import BaseRetriever, TransformRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)

from src.config import LLM_SYSTEM_PROMPT,CHAT_MEMORY_TOKEN_LIMIT
from src.model_loader import initialise_llm

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
    initialise_hyde_llm,
)


def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

    # This reads all the text files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # This breaks the long document into smaller, more manageable chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    # This saves the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """

    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        # We must provide the embed_model when loading the index.
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)


def get_chat_engine(
    llm: Groq,
    embed_model: HuggingFaceEmbedding
) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine with HyDE."""

    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    base_retriever: BaseRetriever = vector_index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K
    )

    # 'include_original=True' ensures the system searches with BOTH
    # the hallucinated answer and the user's real question.
    hyde = HyDEQueryTransform(
        include_original=True,
        llm=llm
    )

    # Now, any query passed to hyde_retriever is first 'imagined' by HyDE.
    hyde_retriever = TransformRetriever(
        retriever=base_retriever,
        query_transform=hyde
    )

    reranker = SentenceTransformerRerank(
        top_n=RERANKER_TOP_N,
        model=RERANKER_MODEL_NAME
    )


    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT,
        llm=llm # potentially you can change this to initialise_hyde_llm() for faster retrieval
    )


    chat_engine = CondensePlusContextChatEngine(
        retriever=hyde_retriever, # Using our new HyDE retriever here
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        node_postprocessors=[reranker]
    )

    return chat_engine


def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""

    print("--- Initialising models... ---")
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: BaseChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model
    )
    print("--- RAG Chatbot Initialised. ---")
    chat_engine.chat_repl()