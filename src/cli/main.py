"""Command-Line Interface for OCR transcription and text redaction."""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

from src.engine.pipeline import OCREngine


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the CLI tool."""
    parser = argparse.ArgumentParser(
        prog="textrecognizer",
        description="Extract text from an image with bounding boxes and produce an inpainting-redacted version.",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the target input image (PNG format required)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to trained PyTorch CRNN model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save output files (defaults to same folder as input image)",
    )
    parser.add_argument(
        "--inpaint-method",
        type=str,
        choices=["telea", "ns"],
        default="telea",
        help="Inpainting algorithm: 'telea' (Alexandru Telea) or 'ns' (Navier-Stokes)",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Execute CLI processing logic."""
    input_path = Path(args.image_path)

    # 1. Validate that input path exists
    if not input_path.exists():
        print(f"Error: File '{input_path}' does not exist.", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"Error: Path '{input_path}' is not a regular file.", file=sys.stderr)
        return 1

    # 2. Validate that the image is a PNG file
    if input_path.suffix.lower() != ".png":
        print(
            f"Error: Invalid file format '{input_path.suffix}'. The application requires PNG images (.png).",
            file=sys.stderr,
        )
        return 1

    # 3. Check model checkpoint
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(
            f"Error: Model checkpoint not found at '{model_path}'. "
            f"Please run training first using: python -m src.training.train",
            file=sys.stderr,
        )
        return 1

    # 4. Initialize OCREngine
    try:
        engine = OCREngine(
            model_path=str(model_path),
            inpaint_method=args.inpaint_method,
        )
    except Exception as e:
        print(f"Error initializing OCR Engine: {e}", file=sys.stderr)
        return 1

    # 5. Process image
    try:
        result = engine.process(input_path, redact=True)
    except Exception as e:
        print(f"Error during OCR and Redaction processing: {e}", file=sys.stderr)
        return 1

    # 6. Save outputs: <stem>_text.json and <stem>_redacted.png
    outputs = engine.save_outputs(
        process_result=result,
        input_path=input_path,
        output_dir=args.output_dir,
    )

    # 7. Print concise processing summary
    num_words = len(result["results"])
    latency = result["latency_ms"]

    print("==================================================")
    print("OCR & TEXT REDACTION COMPLETE")
    print("==================================================")
    print(f"Input:")
    print(f"  {input_path}")
    print(f"Output:")
    print(f"  {outputs['json_path']}")
    print(f"  {outputs['redacted_image_path']}")
    print(f"Summary:")
    print(f"  Detected Words : {num_words}")
    print(f"  Latency        : {latency:.2f} ms")
    print("==================================================")

    if num_words > 0:
        print("Detected Words Sample:")
        for idx, item in enumerate(result["results"][:5], 1):
            print(f"  {idx}. Word: \"{item['word']}\", Bounding Box: {item['bounding_box']}")
        if num_words > 5:
            print(f"  ... and {num_words - 5} more words.")
    print("==================================================")

    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
