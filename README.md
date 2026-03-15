# Project Documentation: StatGuide RAG Auditor

**Author:** Manh Tuong Nguyen
**Domain:** Data Science & Research Methodology
**Target Role:** Data Scientist / ML Engineer / College Students

---

## 1. Project Overview & Use Case

### Problem Statement
In data science and academic research, selecting an inappropriate statistical test leads to "silent failures" and a crisis of reproducibility. Practitioners often ignore critical assumptions—such as normality, homoscedasticity, or independence—due to the dense, non-linear nature of methodology textbooks. This RAG assistant acts as a Methodological Auditor, grounding recommendations in established frameworks to ensure users select the mathematically correct test for their specific data topology.

### Target User
- Data Scientists and Engineering Researchers who need a principled approach to experimental design and a robust way to validate statistical assumptions before reporting results.
- Statistics-related college students who need to study or review statistical concepts.

---

## 2. RAG-Optimized Scholarly Sources

The following 10 documents form the core knowledge base. These are the industry standard guidelines and reputable textbooks in the fields of Statistics that were recommended by professionals.

| # | Source Title | Author | RAG Engineering Alignment |
| :--- | :--- | :--- | :--- |
| 1 | **Design and Analysis of Experiments (9th Ed.)** | Douglas C. Montgomery | Provides sequential protocols for DOE and factor validation. |
| 2 | **NIST Handbook: Exploratory Data Analysis** | NIST (Croarkin & Tobias) | Uses strict sequential numbering (e.g., 1.1.1, 1.1.2) for indexing. |
| 3 | **Comprehensive guidelines for analysis methods** | Jonghae Kim | Features hierarchical headers for decision-making in research. |
| 4 | **Nonparametric Statistical Methods (3rd Ed.)** | Hollander, Wolfe, & Chicken | Explicit definitions of "distribution-free" test logic. |
| 5 | **Probability & Statistics for Engineering and the Sciences** | Jay L. Devore | Sets clear engineering context and defines standard abbreviations. |
| 6 | **NIST Handbook: Process Modeling** | NIST | Clearly defines modeling steps and validation protocols. |
| 7 | **Probability and Statistical Inference** | Hogg, Tanis, & Zimmerman | Provides foundational mathematical proofs for RAG retrieval metrics. |
| 8 | **Testing Statistical Hypotheses** | Erich Lehmann & Joseph Romano | Detailed sections on information criteria (AIC/BIC) and risk. |
| 9 | **Mathematical Statistics: Basic Ideas and Selected Topics** | Bickel & Doksum | Advanced theoretical framework for evaluating estimator consistency. |
| 10 | **Time Series Analysis: Forecasting and Control** | Box, Jenkins, & Reinsel | Critical for modeling RAG performance drift and latency trends. |

---

## 3. System Architecture & Tech Stack

This pipeline was engineered with an **offline-first, privacy-preserving architecture**, running entirely on local compute. It decouples the retrieval, generation, and evaluation layers to allow for independent component optimization.

* **Orchestration Framework:** LangChain combined with custom Python evaluation scripts.
* **Vector Database:** ChromaDB (Local).
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Dense vector representations).
* **Generation LLM:** Gemma 3 (via local Ollama endpoint).
* **Evaluation/Judge LLMs:** DeepSeek-R1 and Llama 3.1 (Used strictly for decoupled, component-level LLM-as-a-judge evaluation).
* **Retrieval Judge:** Gemma 3.

---

## 4. Advanced Retrieval Strategy (Hybrid Search)

Statistical textbooks present a unique retrieval challenge: they require understanding both dense semantic concepts (e.g., "heteroscedasticity") and exact keyword matches (e.g., specific formula names or theorem acronyms). To solve this, the system implements a **Hybrid Search architecture**:

* **Chunking Strategy:** `RecursiveCharacterTextSplitter` with a strict `chunk_size=200` and `chunk_overlap=100`. This high-granularity approach prevents the LLM from getting "lost in the middle" of dense mathematical proofs.
* **Sparse Retrieval (Keyword):** BM25 algorithm (Weight: 0.7, retrieving top k=3).
* **Dense Retrieval (Semantic):** ChromaDB Vector Search (Weight: 0.3, retrieving top k=3).
* **Ensemble:** The results are combined and deduplicated to ensure the generation model receives both conceptual context and exact methodological definitions.

---

## 5. Quantitative Evaluation & Telemetry

To ensure methodological accuracy and prevent statistical hallucinations, the pipeline was subjected to a rigorous **LLM-as-a-judge evaluation loop** across 310 total runs. 

### Generation Quality Metrics
The generation layer (Gemma 3) was evaluated against strict grading rubrics using Llama 3.1 and DeepSeek-R1 as independent judges.

* **Faithfulness (Anti-Hallucination): 92.6%** * *Standard Deviation: 0.184 | Evaluated Runs: 309*
  * *Insight:* The system is highly reliable at grounding its answers strictly in the retrieved textbook context. If the answer is generated, it is mathematically backed by the source material.
* **Valid Reasoning: 83.2%** * *Standard Deviation: 0.374 | Evaluated Runs: 309*
  * *Insight:* The generation model demonstrates strong logical consistency when applying retrieved formulas to the user's specific data topology.

### Retrieval Performance & Telemetry
* **Context Recall:** **36.8%**
* **System Latency:** **3.57s** (Median) | **5.18s** (99th Percentile)
* **Total Token Usage:** 223,642 (184,506 prompt / 39,136 completion)

*Note on System Behavior: The pipeline demonstrates a highly conservative, precision-first behavior. While Context Recall is strict (36.8%), the exceptional Faithfulness score (92.6%) proves that the system successfully identifies when it lacks the proper context and refuses to hallucinate statistical advice.*

---

## 6. Future Roadmap

* **Context Recall Optimization:** Increase the base chunk size from 200 to 500+ characters to capture wider mathematical context, and implement a Hugging Face Cross-Encoder to rerank the hybrid search results.
* **Multi-Vector Summarization:** Decouple the embedded search text from the retrieved context by embedding plain-English summaries of the statistical proofs to bridge the vocabulary gap between user queries and academic text.