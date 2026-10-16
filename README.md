# Hausa Sentiment Analysis — NLP Capstone

Sentiment classification (positive / neutral / negative) for Hausa-language tweets, using the [NaijaSenti](https://github.com/hausanlp/NaijaSenti) dataset and AfriBERTa fine-tuning.

## Project Links
- **GitHub Repository:** *(this repo)*
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/hausa-sentiment-app)
- **Model:** [Hugging Face Hub](https://huggingface.co/YOUR_USERNAME/hausa-sentiment-afriberta)
- **Demo Video:** *(add link after recording)*

---

## Problem
Hausa is spoken by over 70 million people across West Africa, yet NLP resources for it remain limited. This project builds a sentiment classifier for Hausa tweets to support applications in social media monitoring, public health communication, and political analysis.

## Dataset
- **Source:** [NaijaSenti-Twitter (Hausa)](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter) — Muhammad et al. (2022)
- **Size:** ~14,000 tweets (train / validation / test splits provided)
- **Labels:** positive, neutral, negative
- **Language:** Hausa (`hau` config)

## Methodology

| Experiment | Model | LR | Epochs | Test Acc | Test Macro-F1 |
|---|---|---|---|---|---|
| Baseline | TF-IDF (20000, bigram) + Logistic Regression | — | — | 0.7247 | 0.7243 |
| Exp 1 | AfriBERTa-base | 3e-5 | 5 | 0.7839 | 0.7838 |
| Exp 2 | AfriBERTa-base | 1e-5 | 5 | 0.7726 | 0.7709 |
| Exp 3 | XLM-R base | 2e-5 | 5 | 0.7339 | 0.7345 |

## Repository Structure
```
hausa-sentiment/
├── src/
│   ├── data.py          # dataset loading
│   ├── baseline.py      # TF-IDF + Logistic Regression
│   ├── finetune.py      # Transformer fine-tuning
│   └── evaluate.py      # evaluation and error analysis
├── app/
│   ├── app.py           # Streamlit web app
│   └── requirements.txt # Space dependencies
├── hausa_sentiment_afriberta.ipynb  # full Colab notebook
├── requirements.txt
└── README.md
```

## How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run baseline
```bash
cd src
python baseline.py
```

### 3. Fine-tune AfriBERTa (requires GPU — use Google Colab)
```bash
cd src
python finetune.py castorini/afriberta_base 3e-5 5 ../runs/afriberta_3e-5
```

### 4. Evaluate and generate error analysis
```bash
cd src
python evaluate.py
```

### 5. Run the app locally
```bash
cd app
streamlit run app.py
```

## Key Results
AfriBERTa fine-tuned at lr=3e-5 achieves 78.4% accuracy and 0.784 macro-F1, outperforming the TF-IDF baseline by ~6 points. XLM-R base (no Hausa pretraining) scores only 73.4%, confirming that language-specific pretraining is important for Hausa.

## Error Analysis
The neutral class has the lowest F1 (0.73). The most common error is neutral tweets predicted as negative (387 cases) and negative predicted as neutral (240 cases). This reflects the inherent ambiguity between neutral reporting and negative sentiment in Hausa news tweets, as well as code-switching with English and Arabic that the model has not seen enough of during training.

## References
- Muhammad et al. (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis. *LREC 2022*.
- Ogueji et al. (2021). Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages. *MRL 2021*.
- Alabi et al. (2022). Adapting Pretrained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. *COLING 2022*.
- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.


