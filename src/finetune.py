import sys
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
from data import load_splits, label_names

TEXT = "tweet"
model_name = sys.argv[1]
lr = float(sys.argv[2])
epochs = int(sys.argv[3])
out = sys.argv[4]

ds = load_splits()
names = label_names(ds)
tok = AutoTokenizer.from_pretrained(model_name)


def enc(batch):
    return tok(batch[TEXT], truncation=True, max_length=128)


ds = ds.map(enc, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(names),
    id2label=dict(enumerate(names)),
    label2id={n: i for i, n in enumerate(names)},
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "macro_f1": f1_score(p.label_ids, preds, average="macro"),
    }


args = TrainingArguments(
    output_dir=out,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    weight_decay=0.01,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    processing_class=tok,
    compute_metrics=compute_metrics,
)
trainer.train()

test_results = trainer.evaluate(ds["test"])
print("TEST:", test_results)

with open(f"{out}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

trainer.save_model(out)
tok.save_pretrained(out)
