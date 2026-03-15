# Project Documentation: StatGuide RAG Auditor

**Author:** Manh Tuong Nguyen

**Domain:** Data Science & Research Methodology

**Target Role:** Data Scientist / ML Engineer / College Students

---

## 1. Project Overview & Use Case

### Problem Statement

In data science and academic research, selecting an inappropriate statistical test leads to "silent failures" and a crisis of reproducibility. Practitioners
often ignore critical assumptions—such as normality, homoscedasticity, or independence—due to the dense, non-linear nature of methodology textbooks. This RAG assistant acts
as a Methodological Auditor, grounding recommendations in established frameworks to ensure users select the mathematically correct test for their specific data topology.

### Target User

- Data Scientists and Engineering Researchers who need a principled approach to experimental design and a robust way to validate statistical assumptions before reporting results.
- Statistics-related college students who need to study or review statistical concepts.

---

## 2. RAG-Optimized Scholarly Sources

The following 10 documents form the core knowledge base. These are the industry standard guidelines and reputable textbooks in the fields of Statistics that were recommended by 
professionals.

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
