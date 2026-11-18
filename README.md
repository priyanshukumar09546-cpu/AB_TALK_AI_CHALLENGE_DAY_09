# Day 9: Teaching AI Semantic Meaning Through Vectors

**ABTalks 60 Days AI Engineering Coding Challenge**

## Overview
Computers cannot understand raw text directly—they process numbers. In this project, we build a semantic similarity workflow using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization and **Cosine Similarity** to mathematically quantify how closely related different sentences and topics are.

## Objectives Completed
1. **Conceptual Foundations:** Explained why plain text string comparison fails and why numerical vector representations are required for machine processing.
2. **TF-IDF Vectorization:** Converted 10 sentences spanning 3 distinct topics (*Animals, Space, and Cooking*) into numerical vectors using `scikit-learn`'s `TfidfVectorizer`.
3. **Pairwise Cosine Similarity Matrix:** Computed a `10x10` NumPy matrix representing similarity scores across all sentence combinations.
4. **Semantic Search Function (`find_similar`):** Built a custom retrieval function `find_similar(query, corpus, top_k)` to rank and return the most relevant sentences for any input query.
5. **Pairwise Testing:** Evaluated the model on semantically similar pairs (e.g., `'dog'` vs `'puppy'`) and dissimilar pairs (e.g., `'dog'` vs `'airplane'`).
6. **Heatmap Visualization:** Plotted the `10x10` similarity matrix using `matplotlib` with annotated scores and labelled axes to highlight topic clustering.
7. **Vector Math Documentation:** Explained why cosine similarity measures the **angle ($\theta$)** between vectors rather than Euclidean distance, ensuring document length does not distort topic similarity.

## Tech Stack
- **Python 3**
- **NumPy** – Matrix operations and array sorting
- **scikit-learn** – `TfidfVectorizer` and `cosine_similarity`
- **Matplotlib** – Annotated heatmap visualization

## Project Structure
```text
├── Day_09_Semantic_Similarity.ipynb   # Complete Jupyter/Colab notebook with code, outputs, heatmap & explanations
└── README.md                          # Project documentation
