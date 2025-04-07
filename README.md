# LLMSASRec
Basic Prototype of a LLM enhanced SASRec using Pytorch



---

# LLM-Enhanced SASRec for Sequential Recommendation

This repository contains a prototype implementation of a **Sequential Recommendation System** using the [SASRec model](https://arxiv.org/abs/1808.09781), enhanced with **contextual embeddings** extracted from a **pre-trained LLM (BERT-mini)**. The goal is to demonstrate the feasibility of enriching sequential recommendation models with semantic understanding.

---

## Overview

- **Baseline Model**: SASRec (Self-Attentive Sequential Recommendation)
- **Enhancement**: Integration of contextual embeddings from `bert-mini` (a lightweight BERT model)
- **Dataset**: [Amazon Beauty Dataset](https://nijianmo.github.io/amazon/index.html) (user-item interactions, reviews, and metadata)

---



## Features

- SASRec implementation in PyTorch
- Integration of BERT-mini for extracting review-based embeddings
- Fusion of static and contextual item representations
- Leave-One-Out evaluation protocol
- Metrics: NDCG, Hit Rate

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (`pip install transformers`)
- Pandas, NumPy, Scikit-learn

### Run the Notebook

```bash
jupyter notebook llmsasrec.ipynb
```

---

## Evaluation Metrics

- **NDCG (Normalized Discounted Cumulative Gain)**
- **Hit Rate**
---

## Project Structure

```
llmsasrec.ipynb          # Main implementation notebook
data/                    # Processed dataset files
SASRec/                  # LLMSASRec modules
utils.py                 # Helper functions (e.g., metrics, data loaders)
model.py                 # LLMSASRec model definition
main.py                  # Model Training
```

---



##  Acknowledgements

- SASRec: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- BERT: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- Amazon Review Data: [Jianmo Ni et al.](https://nijianmo.github.io/amazon/index.html)

---

## License

MIT License

---

