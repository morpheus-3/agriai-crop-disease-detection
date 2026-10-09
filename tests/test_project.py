import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
import numpy as np
import pytest
from fastapi.testclient import TestClient
from projects import common, agri
from projects.fixtures import generate

@pytest.fixture(scope="session")
def data(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("data"))

@pytest.fixture(scope="session", autouse=True)
def artifact_root(tmp_path_factory):
    previous = common.ROOT
    common.ROOT = tmp_path_factory.mktemp("models")
    from projects import api
    api.ROOT = common.ROOT
    yield common.ROOT
    common.ROOT = previous

@pytest.mark.parametrize("framework",["tensorflow","pytorch"])
def test_agri_training_inference_and_metadata(data, framework):
    report = agri.train(data / "agri.csv",epochs=2,framework=framework,synthetic=True)
    output = agri.predict((data / "leaf_1.png").read_bytes(),"tomato",60,25)
    assert output["label"] in report["labels"]
    assert abs(sum(output["probabilities"].values())-1) < 1e-5
    assert output["review_required"] is True
    with pytest.raises(ValueError):
        agri.predict((data / "leaf_1.png").read_bytes(),"unknown",60,25)
    with pytest.raises(ValueError):
        agri.image(b"not an image")

def test_groq_missing_key_and_provider_contract(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY",raising=False)
    from projects.api import app
    assert TestClient(app).post("/explain",json={"context":{}}).status_code == 503
    captured={}
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":"Model explanation"}}]}
    def post(self,url,**kwargs):
        captured.update(url=url,**kwargs)
        return FakeResponse()
    monkeypatch.setenv("GROQ_API_KEY","test-key-not-real")
    monkeypatch.setattr(common.httpx.Client,"post",post)
    assert common.groq_explain({"score":.5}) == "Model explanation"
    assert captured["url"].startswith("https://api.groq.com/")
    assert captured["headers"]["Authorization"] == "Bearer test-key-not-real"

def test_home_status_and_invalid_input():
    from projects.api import app
    client=TestClient(app)
    assert client.get("/").status_code==200
    assert "AgriAI" in client.get("/").text
    assert client.get("/status").status_code==200
    assert client.post("/agri/predict",files={"file":("x.txt",b"bad","text/plain")},data={"crop":"tomato","humidity":50,"temperature":25}).status_code==415
