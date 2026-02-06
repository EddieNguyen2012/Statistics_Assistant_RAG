# Project Documentation: StatGuide RAG Auditor

**Author:** Manh Tuong Nguyen

**Domain:** Data Science & Research Methodology

**Target Role:** Data Scientist / ML Engineer

---

## 1. Project Overview & Use Case

### Problem Statement

In data science and academic research, selecting an inappropriate statistical test leads to "silent failures" and a crisis of reproducibility. Practitioners often ignore critical assumptions—such as normality, homoscedasticity, or independence—due to the dense, non-linear nature of methodology textbooks. This RAG assistant acts as a Methodological Auditor, grounding recommendations in established frameworks to ensure users select the mathematically correct test for their specific data topology.

### Target User

The target users are Data Scientists and Engineering Researchers who need a principled approach to experimental design and a robust way to validate statistical assumptions before reporting results.

---

## 2. RAG-Optimized Scholarly Sources

The following 10 documents form the core knowledge base. These have been selected for their high authority and structured content, which facilitates the "flat-level syntax" required for high-performance RAG.

| # | Source Title | RAG Engineering Alignment |
| --- | --- | --- |
| 1 | **Design and Analysis of Experiments (9th Ed.)** | Provides sequential protocols for DOE and factor validation. |
| 2 | **Comprehensive guidelines for analysis methods** | Features hierarchical headers for decision-making in research. |
| 3 | **Testing Statistical Assumptions (Garson)** | Organized by specific assumption checks (Normality, etc.). |
| 4 | **Nonparametric Statistical Methods (3rd Ed.)** | Explicit definitions of "distribution-free" test logic. |
| 5 | **NIST Handbook: Exploratory Data Analysis** | Uses strict sequential numbering (e.g., 1.1.1, 1.1.2) for indexing. |
| 6 | **NIST Handbook: Process Modeling** | Clearly defines modeling steps and validation protocols. |
| 7 | **Statistical Hypothesis Testing with SAS and R** | Provides clear "If-Then" logic for test selection. |
| 8 | **Probability & Statistics for Engineering** | Sets clear engineering context and defines standard abbreviations. |
| 9 | **Global Validation of Linear Model Assumptions** | Scholarly focus on disambiguating linear model errors. |
| 10 | **Bayesian Model Selection for Group Studies** | Detailed sections on information criteria (AIC/BIC) and risk. |
