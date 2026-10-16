import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from data import load_splits

ds = load_splits()
tr, va, te = (ds[s].to_pandas() for s in ["train", "validation", "test"])


def run(max_f, ngram, df):
    vec = TfidfVectorizer(max_features=max_f, ngram_range=ngram)
    clf = LogisticRegression(max_iter=1000).fit(
        vec.fit_transform(tr["tweet"]), tr["label"]
    )
    p = clf.predict(vec.transform(df["tweet"]))
    return accuracy_score(df["label"], p), f1_score(df["label"], p, average="macro")


rows = []
for mf in [2000, 5000, 20000]:
    for ng in [(1, 1), (1, 2)]:
        a, f = run(mf, ng, va)
        rows.append(dict(max_features=mf, ngram=str(ng), val_acc=round(a, 4), val_f1=round(f, 4)))

g = pd.DataFrame(rows)
print(g.to_string(index=False))

best = g.sort_values("val_f1").iloc[-1]
a, f = run(int(best.max_features), eval(best.ngram), te)
print(f"\nBest config: max_features={int(best.max_features)}, ngram={best.ngram}")
print(f"TEST accuracy={a:.4f}  macro_f1={f:.4f}")
