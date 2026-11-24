from fastapi.testclient import TestClient
from app import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_positive():
    r = c.post("/predict", json={"text": "fantastic quality, I love it"}).json()
    assert r["label"] == "positive"


def test_negative():
    r = c.post("/predict", json={"text": "terrible, a waste of money"}).json()
    assert r["label"] == "negative"


def test_validation_error():
    assert c.post("/predict", json={}).status_code == 422
