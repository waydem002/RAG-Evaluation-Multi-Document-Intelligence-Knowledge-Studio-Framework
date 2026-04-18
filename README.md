# RAG Evaluation Multi-Document Intelligence & Knowledge Studio Framework

This directory houses a standalone **"Test-Lab"** designed to ruthlessly evaluate your RAG application's quality using the **Ragas** metric framework, alongside the newly integrated **Multimodal Knowledge Studio**. We evaluate models quantitatively to ensure mathematical adherence rather than feeling-based chat responses.

## Core Metrics Graded

The evaluation harness generates a scorecard natively inside `evaluation_results/` that calculates:
* **Faithfulness:** Is the LLM hallucinating? Can you trust its answers? (Determined by ensuring the generated answer is derived explicitly from the retrieved context).
* **Answer Correctness:** Does the LLM perfectly match the factual semantics of the true answer we provided in our ground-truths?
* **Context Precision:** How much noise is the retriever sending? High precision means the chunks sent to the LLM are dense with relevance instead of fluff.
* **Context Recall:** Out of all the information required to perfectly answer the question, what percentage of it did the retriever successfully locate?

---

## 🎙️ Multimodal Knowledge Studio

The system now features a **Production-Grade Knowledge Studio** that transforms static RAG retrievals into an immersive audio experience. This bridges the gap between raw data and accessible insights.

* **Dialogue Engineering:** A specialized generator that transforms technical RAG responses into natural, long-form podcast scripts between two hosts, **Sarah and Maya**.
* **Context-Aware Synthesis:** Uses the `llama-3.3-70b-versatile` model to ensure the podcast remains factually grounded in the retrieved PDF data.
* **Zero-Latency Audio Pipeline:** Utilizes a Base64-encoded audio stream to deliver synthesized speech instantly to the browser, bypassing file-system bottlenecks and "Disk-Lock" errors.
* **Human-in-the-Loop:** Features an interactive Script Editor, allowing users to manually refine or extend the AI-generated dialogue before final audio synthesis.

---

## How It Works

Because making evaluation calls to an external API like Groq is volatile and generally breaks free-tier Rate Limits rapidly, our engine separates operations cleanly:
1.  **Model Separation:** We initialize a **completely separate LLM and Evaluation Embedder model** (`llama-3.3-70b-versatile` and `BAAI/bge-large-en-v1.5`) via `evaluation_model_loader.py`.
2.  **Independent Storage:** It dynamically generates its own **scratch-pad** Vector Store caching indices (`local_storage/evaluation_vector_stores`) independent of production.
3.  **Extended Output Limits:** Configured with an increased `LLM_MAX_NEW_TOKENS` (1024) to support long-form narrative generation (up to 120+ seconds of audio).
4.  **Graceful Processing:** All inquiries are systematically processed incrementally with a `SLEEP_PER_EVALUATION` pause to bypass tokens-per-minute constraints.

---

## Evaluation Stages

The engine now supports a five-stage optimization process:
* **Stage 1: Baseline Evaluation:** Measures performance using current parameters in `src/config.py`.
* **Stage 2: Chunking Strategy:** Systematically tests multiple chunking configurations to identify the "Goldilocks zone" for your specific data.
* **Stage 3: Reranker Strategy:** Evaluates the impact of adding a cross-encoder reranker stage, testing various `retriever_k` and `reranker_n` combinations.
* **Stage 4: Query Rewriter (HyDE):** Measures the impact of Hypothetical Document Embeddings on retrieval performance.
* **Stage 5: Synthesis Quality:** Evaluates the narrative flow and factual consistency of the Knowledge Studio outputs.

---

## 🚀 Future Roadmap (V3.0)

I am currently developing the following enhancements to push the boundaries of this RAG framework:
* **Advanced Multi-Voice Synthesis:** Integration with high-fidelity TTS providers (ElevenLabs/OpenAI) to provide distinct, human-like voices for Sarah and Maya.
* **Automated Video Briefings:** Merging synthesized audio with dynamic PowerPoint-style slides or AI-generated avatars to create full video reports.
* **Graph-RAG Integration:** Transitioning to Knowledge Graphs to allow the podcast hosts to discuss complex relationships between concepts across thousands of documents.
* **Real-time Streaming TTS:** Reducing synthesis wait time for 5+ minute podcasts by streaming audio chunks to the player as they are generated.

---

## Usage

You do **NOT** execute anything inside this directory. To run an evaluation loop, simply execute the `evaluate.py` script located at the root of your project:
```bash
python evaluate.py
```

To use the Knowledge Studio, run the main application and navigate to the **Knowledge Studio Card** after receiving a chat response:
```bash
streamlit run app.py
```

## Customizing The Experiments

To define new chunking strategies to test:
1.  Open `evaluation/evaluation_config.py`.
2.  Update the `CHUNKING_STRATEGY_CONFIGS` list with your desired `size` and `overlap` pairs.

## Customizing The Ground-Truths

When you delete `python notes*(1)(1).pdf` from your production `/data` folder to inject your own RAG data, you **MUST** update the test harness!
1.  Edit `evaluation_questions.py`.
2.  Formulate 5-6 hard questions explicitly pertaining to the knowledge you just added.
3.  Draft the exact **factual ground_truth** answer the system must achieve.
4.  Run `python evaluate.py` and review your `.csv` report in `evaluation_results/`.