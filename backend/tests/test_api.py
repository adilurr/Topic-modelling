"""
Unit tests for the Topic Modeling Demo API.
Run with:  cd backend && python -m pytest tests/ -v
"""

import io
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Health & Meta
# ---------------------------------------------------------------------------

def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_presets_returns_four_items():
    res = client.get("/presets")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4
    for key in ("tech", "economics", "climate", "health"):
        assert key in data
        assert "title" in data[key]
        assert "text" in data[key]


# ---------------------------------------------------------------------------
# Suggest Topics
# ---------------------------------------------------------------------------

def test_suggest_topics_short_text():
    res = client.get("/suggest-topics", params={"text": "Hello world this is a test sentence."})
    assert res.status_code == 200
    data = res.json()
    assert data["suggested_num_topics"] == 2
    assert "reason" in data


def test_suggest_topics_rich_text():
    long_text = (
        "Artificial intelligence and machine learning models require large computational resources. "
        "Deep neural networks perform representation learning on complex structured data. "
        "Cloud infrastructure platforms like AWS and Azure allow developers to deploy AI services. "
        "Cybersecurity experts monitor global network traffic to protect enterprise systems. "
        "Data engineering pipelines clean and aggregate telemetry for predictive models. "
        "DevOps automation speeds up software delivery cycles across hybrid cloud networks. "
        "Renewable energy generation from solar farms and wind turbines is growing rapidly. "
        "Climate scientists emphasize urgent reductions in carbon emissions worldwide. "
        "Central banks adjusted interest rates to control rising inflation this quarter. "
        "Venture capital funds are backing early stage technology startups globally."
    )
    res = client.get("/suggest-topics", params={"text": long_text})
    assert res.status_code == 200
    data = res.json()
    assert 2 <= data["suggested_num_topics"] <= 8


def test_suggest_topics_missing_param():
    res = client.get("/suggest-topics")
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Predict — LDA
# ---------------------------------------------------------------------------

GOOD_TEXT = (
    "Artificial intelligence and machine learning models require large computational resources. "
    "Deep neural networks perform representation learning on complex structured data. "
    "Cloud infrastructure platforms like AWS and Azure deploy scalable AI microservices. "
    "Cybersecurity experts monitor global network traffic to protect enterprise systems. "
    "Data engineering pipelines clean and aggregate telemetry for predictive models. "
    "DevOps automation speeds up software delivery cycles across hybrid cloud networks."
)


def test_predict_lda_success():
    res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 3,
        "method": "lda",
        "use_bigrams": True,
        "max_words_per_topic": 8,
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["topics"]) == 3
    assert data["method_used"] == "LDA"
    for topic in data["topics"]:
        assert "label" in topic
        assert "top_words" in topic
        assert "percentage" in topic
        assert isinstance(topic["label"], str)
        assert len(topic["top_words"]) > 0
    for seg in data["segments"]:
        assert "topic_label" in seg
        assert 0.0 <= seg["confidence"] <= 1.0


def test_predict_lda_stats():
    res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 3,
        "method": "lda",
        "use_bigrams": False,
        "max_words_per_topic": 5,
    })
    assert res.status_code == 200
    data = res.json()
    stats = data["stats"]
    assert stats["total_words"] > 0
    assert stats["vocab_size"] > 0
    assert stats["total_sentences"] >= 3


# ---------------------------------------------------------------------------
# Predict — NMF
# ---------------------------------------------------------------------------

def test_predict_nmf_success():
    res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 2,
        "method": "nmf",
        "use_bigrams": True,
        "max_words_per_topic": 6,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["method_used"] == "NMF"
    assert len(data["topics"]) == 2


# ---------------------------------------------------------------------------
# Predict — Validation Errors
# ---------------------------------------------------------------------------

def test_predict_empty_text():
    res = client.post("/predict", json={
        "text": "",
        "num_topics": 3,
        "method": "lda",
    })
    assert res.status_code == 400


def test_predict_too_short_for_topics():
    res = client.post("/predict", json={
        "text": "Short text only.",
        "num_topics": 5,
        "method": "lda",
    })
    assert res.status_code in (400, 422)


def test_predict_invalid_num_topics_too_low():
    res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 1,   # below minimum of 2
        "method": "lda",
    })
    assert res.status_code == 422


def test_predict_invalid_num_topics_too_high():
    res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 10,  # above maximum of 8
        "method": "lda",
    })
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------

def test_upload_plain_text():
    content = b"This is a plain text document for topic modeling tests."
    res = client.post(
        "/upload-file",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "extracted_text" in data
    assert len(data["extracted_text"]) > 0
    assert data["filename"] == "test.txt"


def test_upload_empty_file():
    res = client.post(
        "/upload-file",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert res.status_code == 400


def test_upload_csv():
    csv_content = b"topic,description\nmachine learning,AI research\ncloud,infrastructure"
    res = client.post(
        "/upload-file",
        files={"file": ("data.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["extracted_text"]) > 0


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def test_export_csv_returns_csv():
    # First get results
    predict_res = client.post("/predict", json={
        "text": GOOD_TEXT,
        "num_topics": 3,
        "method": "lda",
    })
    assert predict_res.status_code == 200

    # Then export
    export_res = client.post("/export-csv", json=predict_res.json())
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers.get("content-type", "")
    assert len(export_res.text) > 0
    assert "TOPIC SUMMARY" in export_res.text
    assert "SEGMENT ASSIGNMENTS" in export_res.text
