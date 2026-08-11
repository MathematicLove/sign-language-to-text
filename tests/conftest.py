import pytest

@pytest.fixture(autouse=True)
def no_model_download(monkeypatch):
    monkeypatch.setenv("SLT_NO_DOWNLOAD", "1")