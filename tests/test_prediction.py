"""Integration tests for FastAPI endpoints and prediction services."""

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.config.settings import settings

# Disable RabbitMQ background thread during testing
settings.RABBITMQ_ENABLED = False


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "DebtLens ML Service"
    assert data["status"] == "RUNNING"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["models"]["satd_model_loaded"] is True
    assert data["models"]["bug_model_loaded"] is True


def test_predict_satd_comment(client):
    payload = {"comment": "// TODO: this is a temporary workaround for database connection timeout"}
    response = client.post("/api/v1/predict/satd", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidenceScore" in data
    assert 0.0 <= data["confidenceScore"] <= 1.0
    assert "isDebt" in data


def test_predict_class(client):
    payload = {
        "className": "com.debtlens.analysisservice.TestClass",
        "filePath": "src/main/java/TestClass.java",
        "startLine": 1,
        "endLine": 45,
        "comments": [
            "// FIXME: potential null pointer exception when cache is empty",
            "// Normal comment explaining business logic"
        ],
        "cbo": 4,
        "dit": 2,
        "fanIn": 3,
        "fanOut": 2,
        "lcom": 12.0,
        "noc": 0,
        "numberOfAttributes": 5,
        "numberOfLinesOfCode": 45,
        "numberOfMethods": 6,
        "numberOfPrivateAttributes": 4,
        "numberOfPrivateMethods": 3,
        "numberOfPublicAttributes": 1,
        "numberOfPublicMethods": 3,
        "rfc": 14,
        "wmc": 8.0,
        "numberOfVersionsUntil": 5,
        "numberOfAuthorsUntil": 2,
        "linesAddedUntil": 45,
        "maxLinesAddedUntil": 30,
        "avgLinesAddedUntil": 9.0,
        "linesRemovedUntil": 10,
        "maxLinesRemovedUntil": 5,
        "avgLinesRemovedUntil": 2.0,
        "codeChurnUntil": 35,
        "maxCodeChurnUntil": 25,
        "avgCodeChurnUntil": 7.0,
        "ageWithRespectTo": 12.5,
        "weightedAgeWithRespectTo": 4.2
    }
    response = client.post("/api/v1/predict/class", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["className"] == "com.debtlens.analysisservice.TestClass"
    assert "bugPrediction" in data
    assert "probabilityScore" in data["bugPrediction"]
    assert 0.0 <= data["bugPrediction"]["probabilityScore"] <= 1.0
    assert "prediction" in data["bugPrediction"]
    assert "satdDetections" in data
    assert len(data["satdDetections"]) == 2


def test_predict_batch_job(client):
    payload = {
        "jobId": "analysis-job-999",
        "repositoryId": "repo-42",
        "classes": [
            {
                "className": "UserService",
                "filePath": "src/main/java/UserService.java",
                "startLine": 1,
                "endLine": 100,
                "comments": ["// TODO: refactor SQL query"],
                "cbo": 2,
                "dit": 1,
                "fanIn": 1,
                "fanOut": 2,
                "lcom": 4.0,
                "noc": 0,
                "numberOfAttributes": 3,
                "numberOfLinesOfCode": 100,
                "numberOfMethods": 5,
                "numberOfPrivateAttributes": 2,
                "numberOfPrivateMethods": 2,
                "numberOfPublicAttributes": 1,
                "numberOfPublicMethods": 3,
                "rfc": 10,
                "wmc": 5.0,
                "numberOfVersionsUntil": 4,
                "numberOfAuthorsUntil": 2,
                "linesAddedUntil": 100,
                "maxLinesAddedUntil": 80,
                "avgLinesAddedUntil": 25.0,
                "linesRemovedUntil": 20,
                "maxLinesRemovedUntil": 15,
                "avgLinesRemovedUntil": 5.0,
                "codeChurnUntil": 80,
                "maxCodeChurnUntil": 65,
                "avgCodeChurnUntil": 20.0,
                "ageWithRespectTo": 6.0,
                "weightedAgeWithRespectTo": 2.0
            }
        ]
    }
    response = client.post("/api/v1/predict/job", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["jobId"] == "analysis-job-999"
    assert data["totalClassesAnalyzed"] == 1
    assert data["totalCommentsClassified"] == 1
    assert len(data["classes"]) == 1
