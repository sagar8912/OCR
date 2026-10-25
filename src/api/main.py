"""FastAPI web service exposing OCR and text redaction inference engine with interactive web dashboard."""

import os
import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.engine.pipeline import OCREngine

# Global reference for engine instance loaded on startup
app_engine: Optional[OCREngine] = None
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads the OCR model once on server startup."""
    global app_engine
    model_path = os.getenv("OCR_MODEL_PATH", "models/best_model.pt")
    print(f"[FastAPI Startup] Initializing OCREngine with model: {model_path}...")
    try:
        app_engine = OCREngine(model_path=model_path)
        app.state.engine = app_engine
        print("[FastAPI Startup] OCREngine initialized and ready for CPU inference.")
    except Exception as e:
        print(f"[FastAPI Startup Error] Could not load model: {e}")
        app_engine = None
        app.state.engine = None
    yield
    print("[FastAPI Shutdown] Shutting down OCR service.")


app = FastAPI(
    title="OCR & Text Redaction Engine API",
    description="Production-grade CPU-optimized Optical Character Recognition and Inpainting-based Redaction Service",
    version="1.0.0",
    lifespan=lifespan,
)


class WordDetection(BaseModel):
    word: str = Field(..., description="Recognized word transcription")
    bounding_box: List[int] = Field(
        ...,
        description="Word bounds in format [top, left, bottom, right] (ymin, xmin, ymax, xmax)",
        json_schema_extra={"example": [0, 10, 15, 100]},
    )
    confidence: Optional[float] = Field(
        None,
        description="Recognition confidence score in [0.0, 1.0]",
        json_schema_extra={"example": 0.95},
    )


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


@app.get("/", response_class=HTMLResponse, tags=["Web Dashboard"], summary="Interactive Visual Frontend")
async def serve_frontend():
    """Serves the interactive web visualizer dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse("<h2>OCR Studio Frontend ready. Please place index.html in src/api/static.</h2>")


@app.get("/sample-image", tags=["Web Dashboard"], summary="Retrieve sample demo images")
async def get_sample_image(name: str = Query("document", description="'document' or 'word'")):
    """Returns a sample image for quick interactive demonstration."""
    if name == "word":
        sample_path = Path("IIIT5K-Word_V3.0/IIIT5K/test/1002_2.png")
    elif name == "invoice":
        sample_path = Path("sample/invoice_test.png")
    else:
        sample_path = Path("sample/document_sample.png")

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found on disk.")
    return FileResponse(sample_path, media_type="image/png")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint confirming API service and model availability."""
    is_ready = app_engine is not None and app_engine.recognizer.model is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
    }


@app.post(
    "/ocr",
    response_model=List[WordDetection],
    tags=["OCR"],
    summary="Extract text words and bounding boxes from an image",
)
async def process_ocr(
    file: UploadFile = File(..., description="Image file to analyze (PNG preferred)"),
):
    """Extract recognized words and their exact bounding boxes [top, left, bottom, right].

    - **file**: Multipart uploaded image file.
    - **Returns**: JSON list of detected words with coordinates.
    """
    if app_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR Engine is not initialized or model checkpoint was not found.",
        )

    filename = file.filename or "unknown.png"
    if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{filename}'. Please upload a PNG or JPEG image.",
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        result = app_engine.process(file_bytes, redact=False)
        return result["results"]

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {str(e)}",
        )


@app.post(
    "/process",
    tags=["Web Dashboard"],
    summary="Full pipeline: OCR transcription + inpainting redaction with base64 visualizer output",
)
async def process_full_pipeline(
    file: UploadFile = File(..., description="Image file to process"),
):
    """Process an image and return word transcriptions, bounding boxes, and base64 redacted image."""
    if app_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR Engine is not initialized.",
        )

    try:
        file_bytes = await file.read()
        result = app_engine.process(file_bytes, redact=True)

        redacted_b64 = None
        redacted_img = result.get("redacted_image")
        if redacted_img is not None:
            _, buffer = cv2.imencode(".png", redacted_img)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            redacted_b64 = f"data:image/png;base64,{b64_str}"

        return {
            "results": result["results"],
            "redacted_image_base64": redacted_b64,
            "latency_ms": result["latency_ms"],
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post(
    "/redact",
    tags=["Redaction"],
    summary="Upload image and receive inpainting-redacted PNG image file",
)
async def process_redact(
    file: UploadFile = File(..., description="Image file to redact"),
):
    """Detect text regions in the uploaded image and return the redacted image (PNG)."""
    if app_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR Engine is not initialized.",
        )

    try:
        file_bytes = await file.read()
        result = app_engine.process(file_bytes, redact=True)
        redacted_img = result.get("redacted_image")

        if redacted_img is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate redacted image.",
            )

        success, encoded_png = cv2.imencode(".png", redacted_img)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode redacted image to PNG.",
            )

        return Response(
            content=encoded_png.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=image_redacted.png"},
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Redaction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
