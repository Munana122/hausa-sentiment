import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from data import load_splits

MODEL_PATH = "models/afriberta_best"
TEXT = "tweet"

ds = load_splits()
te = ds["test"].to_pandas()

clf = pipeline("text-classification", model=MODEL_PATH, truncation=True, device=-1)

te["pred_label"] = [clf(t)[0]["label"] for t in te[TEXT]]
te["true_label"] = te["label"].map(clf.model.config.id2label)

print(classification_report(te["true_label"], te["pred_label"]))

labels = list(clf.model.config.id2label.values())
cm = confusion_matrix(te["true_label"], te["pred_label"], labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

errors = te[te["true_label"] != te["pred_label"]]
correct = te[te["true_label"] == te["pred_label"]]

errors.sample(min(40, len(errors)), random_state=1).to_csv("errors_sample.csv", index=False)
correct.sample(min(20, len(correct)), random_state=1).to_csv("correct_sample.csv", index=False)
