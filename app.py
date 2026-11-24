"""Tiny sentiment API: TF-IDF + logistic regression trained at startup on a small hand-written dataset."""
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TRAIN = [
    ("I love this product, it works great", 1), ("Absolutely fantastic service", 1),
    ("Really happy with the quality", 1), ("Excellent value and fast delivery", 1),
    ("Wonderful experience, would buy again", 1), ("The support team was helpful and kind", 1),
    ("This is terrible, it broke after one day", 0), ("Awful quality and rude staff", 0),
    ("I am very disappointed", 0), ("Worst purchase I have ever made", 0),
    ("Slow delivery and poor packaging", 0), ("It does not work, a complete waste of money", 0),
]

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
model.fit([t for t, _ in TRAIN], [y for _, y in TRAIN])

app = FastAPI(title="Sentiment API")


class Item(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Item):
    p = float(model.predict_proba([item.text])[0][1])
    return {"label": "positive" if p >= 0.5 else "negative", "positive_probability": round(p, 4)}
