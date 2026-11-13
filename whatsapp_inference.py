"""CLI bridge that reuses the Streamlit app's YOLO segmentation pipeline."""

import json
import sys
from pathlib import Path

from PIL import Image

from app import analyze_image


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python whatsapp_inference.py INPUT OUTPUT OVERLAY_METADATA FILENAME"
        )

    input_path, output_path, metadata_path, filename = sys.argv[1:]
    image = Image.open(input_path).convert("RGB")
    (
        overlay,
        tumor_type,
        region_count,
        confidence,
        mask_area,
        explanation,
        _probabilities,
        regions,
    ) = analyze_image(image, filename)
    Image.fromarray(overlay, mode="RGB").save(output_path, format="PNG")
    Path(metadata_path).write_text(
        json.dumps(
            {
                "tumorType": tumor_type,
                "regionCount": region_count,
                "confidence": confidence,
                "maskArea": mask_area,
                "explanation": explanation,
                "regions": regions,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
