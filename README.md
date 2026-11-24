# fastapi-sentiment-api

A minimal FastAPI service with `/health` and `/predict`. A TF-IDF + logistic regression model is trained at startup on 12 hand-written sentences. It is a template for serving a scikit-learn model, not an accurate sentiment model.

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "fantastic quality"}'
pytest
```

## Output (reproduced through FastAPI's TestClient)

```
fantastic quality, I love it -> {'label': 'positive', 'positive_probability': 0.5348}
terrible, a waste of money -> {'label': 'negative', 'positive_probability': 0.4355}

$ pytest -q
4 passed, 1 warning in 0.68s
```

## Limits

- Training data is tiny, so probabilities stay close to 0.5. Swap `TRAIN` for a real dataset before relying on predictions.
