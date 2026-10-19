# Hausa Tweet Sentiment Classification

A sentiment classifier (**positive/neutral/negative**) for Hausa tweets. I fine-tuned **AfriBERTa** on the Hausa part of **NaijaSenti**, compared it with a **TF-IDF baseline** and **XLM-R base**, and put the best model behind a small web app.

This is my individual capstone project for Machine Learning Techniques I (ALU, 2026 September term), task type: **text classification for an African language**.

## Links

- **Live app (Hugging Face Space):**  https://github.com/Munana122/hausa-sentiment
- **Fine-tuned model:** https://huggingface.co/Merveille123/hausa-sentiment-afriberta
- **Demo video:** https://youtu.be/9q0DRn-ZsLI

## Results at a glance

| Model | Learning rate | Test accuracy | Test macro-F1 |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | n/a | 0.7290 | 0.7291 |
| **AfriBERTa base** | 3e-5 | 0.7660 | 0.7632 |
| **AfriBERTa base** | 1e-5 | 0.7718 | 0.7704 |
| XLM-R base | 2e-5 | 0.7222 | 0.7221 |

- **Fine-tuning AfriBERTa beat the baseline by about 3.4 to 4.1 macro-F1 points.**
- **The two learning rates are effectively tied.** In earlier sessions, repeated runs of the 3e-5 setting gave test macro-F1 anywhere from 0.762 to 0.784, so differences under about two points should be read as noise.
- **XLM-R base did not clearly beat the TF-IDF baseline.**
- Each configuration in the final notebook was trained **once with seed 42**.

## The problem

Hausa is one of the four most widely spoken languages in Nigeria (Muhammad et al., 2022), but there are few tools for analysing it. A sentiment classifier lets someone follow public opinion in Hausa tweets without reading each one. I chose Hausa because NaijaSenti gives a human-labelled Hausa dataset with fixed splits, and because AfriBERTa (Ogueji et al., 2021) was pretrained on Hausa, so I could compare it with a broader multilingual model.

I set out to answer three questions:

1. Does fine-tuning a Transformer beat a TF-IDF + logistic regression baseline?
2. Does a model pretrained on African languages (AfriBERTa) beat a broad multilingual one (XLM-R base)?
3. How much do results move with the learning rate and between runs, and where does the best model fail?

## Dataset

- **Source:** NaijaSenti-Twitter, Hausa (`hau`) configuration (Muhammad et al., 2022), loaded from the Hugging Face Hub (`HausaNLP/NaijaSenti-Twitter`).
- **Licence:** CC BY-NC-SA 4.0, so it is for **non-commercial use only**.
- **Size:** **22,152** human-labelled tweets, using the splits the authors provide.

| Split | Tweets | Positive | Neutral | Negative |
| --- | --- | --- | --- | --- |
| Train | 14,172 | 4,687 | 4,912 | 4,573 |
| Validation | 2,677 | 887 | 896 | 894 |
| Test | 5,303 | 1,755 | 1,789 | 1,759 |

The classes are balanced (32 to 35% each), so accuracy is not distorted by a big majority class.

**Preprocessing.** I kept it minimal. The baseline uses scikit-learn's default tokenisation and lowercasing. The Transformers use their own SentencePiece tokenisers with truncation at 128 tokens. I left handles (`@user`), emojis and spelling alone, because they may carry sentiment and I did not want to impose my own rules on Hausa spelling.

**Known problems with the data.** Tweets are informal, spelling varies, and English words are mixed in. The **neutral label is the most subjective one**. I did not measure annotator agreement, and I did not check for duplicate tweets across splits.

## Methods

**Baseline.** TF-IDF features plus logistic regression (scikit-learn). I tried `max_features` in {2,000, 5,000, 20,000} and n-grams (1,1) or (1,2), picked the best setting on the **validation** split (20,000 features, unigrams and bigrams) and evaluated it **once** on the test split.

**Fine-tuned models.** Both are encoder-only Transformers:

- **AfriBERTa base** (`castorini/afriberta_base`): 8 layers, about 111M parameters, pretrained on 11 African languages including Hausa.
- **XLM-R base** (`xlm-roberta-base`): pretrained on 100 languages (Hausa is one of them, with much less data than the big languages).

Each tweet is tokenised and passed through the encoder. A classification head on the first-token vector (dense layer, tanh, dropout, linear layer) outputs three scores. The head starts from random weights, and the whole network is fine-tuned with cross-entropy loss.

**Training settings** (same for every run except the learning rate): AdamW with weight decay 0.01, linear learning-rate decay, batch size 16, 5 epochs, max length 128, seed 42, Hugging Face `Trainer` on a free Colab T4 GPU. After each epoch, I evaluated on the validation split and kept the checkpoint with the **best validation macro-F1**.

**Metrics.** Accuracy and **macro-F1** on the test set, plus per-class precision and recall. Macro-F1 is my main metric because it gives every class equal weight, so a weak class like neutral cannot hide.

### Baseline search (validation split)

| max_features | n-grams | Val accuracy | Val macro-F1 |
| --- | --- | --- | --- |
| 2,000 | (1,1) | 0.7340 | 0.7354 |
| 2,000 | (1,2) | 0.7366 | 0.7383 |
| 5,000 | (1,1) | 0.7434 | 0.7445 |
| 5,000 | (1,2) | 0.7486 | 0.7501 |
| 20,000 | (1,1) | 0.7422 | 0.7435 |
| **20,000** | **(1,2)** | **0.7527** | **0.7548** |

## Error analysis

I analysed the AfriBERTa model trained with a learning rate of 3e-5 (the one I deployed). It makes **1,240 errors** on 5,303 test tweets.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Positive | 0.85 | 0.86 | 0.85 | 1,755 |
| **Neutral** | 0.79 | **0.60** | **0.68** | 1,789 |
| Negative | 0.69 | 0.85 | 0.76 | 1,759 |

Confusion matrix (rows are the true labels):

| True \ Predicted | Positive | Neutral | Negative |
| --- | --- | --- | --- |
| Positive | 1,504 | 131 | 120 |
| Neutral | 159 | 1,071 | **559** |
| Negative | 115 | 156 | 1,488 |

- **The main mistake is neutral tweets predicted as negative:** 559 of them, which is **31% of all neutral tweets and 45% of all errors**.
- **The model leans negative.** It predicts negative 2,167 times, but only 1,759 test tweets really are negative.
- **Positive is the cleanest class.**

I have ideas about why, but I have **not tested them**. The line between neutral and negative is subjective in tweets (a report of bad news, or a mild complaint, can be labelled either way), so some of these errors may be label noise rather than model mistakes. Mixed English and informal spelling probably give the model fewer reliable clues. Checking this properly would need a Hausa speaker to re-label a sample of the errors.

## Limitations

- One platform (Twitter), one language, three classes.
- **One run per setting** in the final notebook, no confidence intervals or significance tests, and the learning-rate comparison is inside the run-to-run noise.
- Only two learning rates for AfriBERTa and one for XLM-R, with no wider search and no class weighting.
- The probabilities shown in the app have **not been checked for calibration**, so do not read them as exact confidence.
- The app has **no language check**: any text, even English, gets a Hausa sentiment label.
- No Hausa speaker has verified my reading of the errors.

## Repository contents

```
hausa-sentiment/
├── hausa_sentiment_afriberta.ipynb   # full workflow: data, baseline, fine-tuning, evaluation (reference for the numbers above)
├── src/                              # command-line versions of the same steps
│   ├── data.py
│   ├── baseline.py
│   ├── finetune.py
│   └── evaluate.py
├── app/
│   ├── app.py                        # Streamlit app
│   └── requirements.txt              # dependencies for the Hugging Face Space
├── requirements.txt
└── README.md
```

## How to run it

1. Install the dependencies: `pip install -r requirements.txt`
2. To redo the experiments, open `hausa_sentiment_afriberta.ipynb` in **Google Colab with a GPU** (Runtime > Change runtime type > T4 GPU) and run the cells in order. Fine-tuning needs a GPU. The baseline runs fine on a CPU.
3. To run the web app on your own machine: `python -m streamlit run app/app.py`. The first prediction downloads the model, so it is slow.

Training is not fully deterministic on GPU, so a rerun can differ from my numbers by a point or two.

## Resources and credits

- **Data:** NaijaSenti (Muhammad et al., 2022), CC BY-NC-SA 4.0.
- **Pretrained models:** AfriBERTa (Ogueji et al., 2021) and XLM-R (Conneau et al., 2020).
- **Libraries:** scikit-learn, Hugging Face Transformers and Datasets, PyTorch, Streamlit.
- **What I did myself:** ran the baseline search, set up and ran the fine-tuning experiments, evaluated the models, analysed the errors, and built and deployed the app.
- **AI assistance:** AI assistants (Gemini and Claude) helped me with code and with drafting the write-up. I ran and tested everything, and I am responsible for the results reported here.

## References

- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. COLING 2022, 4336-4349.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. ACL 2020, 8440-8451.
- Lhoest, Q., et al. (2021). Datasets: A community library for natural language processing. EMNLP 2021 (System Demonstrations), 175-184.
- Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter sentiment corpus for multilingual sentiment analysis. LREC 2022, 590-602.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. MRL 2021, 116-126.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS 32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12, 2825-2830.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. EMNLP 2020 (System Demonstrations), 38-45.
