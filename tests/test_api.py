"""Integration tests for FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_dummy_png_bytes() -> bytes:
    """Helper to generate a minimal valid PNG image in-memory."""
    img = Image.new("RGB", (100, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data


def test_ocr_endpoint_valid_image(client):
    png_bytes = create_dummy_png_bytes()
    response = client.post(
        "/ocr",
        files={"file": ("test.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "word" in data[0]
        assert "bounding_box" in data[0]
        assert len(data[0]["bounding_box"]) == 4


def test_ocr_endpoint_invalid_extension(client):
    response = client.post(
        "/ocr",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_redact_endpoint_returns_png(client):
    png_bytes = create_dummy_png_bytes()
    response = client.post(
        "/redact",
        files={"file": ("test.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_ocr_endpoint_empty_file(client):
    response = client.post(
        "/ocr",
        files={"file": ("test.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ocr_endpoint_corrupted_file(client):
    response = client.post(
        "/ocr",
        files={"file": ("corrupt.png", b"not a real png header or data", "image/png")},
    )
    assert response.status_code == 400
    assert "invalid or corrupted" in response.json()["detail"].lower()

