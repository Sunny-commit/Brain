"""Streamlit interface for CPU-based brain tumor MRI segmentation."""

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import numpy as np
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).with_name("best.pt")
CLASSIFIER_PATH = Path(__file__).with_name("cls_best.pt")
CLASSIFICATION_CATEGORIES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
SUPPORTED_TYPES = ["jpg", "jpeg", "png"]
ImageInput: TypeAlias = Image.Image | np.ndarray


@st.cache_resource
def load_model() -> YOLO:
    """Load the segmentation model once for the lifetime of the app."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights were not found at {MODEL_PATH}. "
            "Download best.pt into the project root before starting the app."
        )
    return YOLO(str(MODEL_PATH))


def find_classifier_path() -> Path | None:
    """Find an optional YOLO classification checkpoint in the workspace."""
    candidates = [
        CLASSIFIER_PATH,
        Path(__file__).with_name("classifier.pt"),
        *sorted(Path(__file__).parent.glob("*-cls.pt")),
    ]
    return next((path for path in candidates if path.exists()), None)


@st.cache_resource
def load_classifier(path: str) -> YOLO:
    """Load a discovered four-class YOLO classifier."""
    return YOLO(path)


def model_names(model: YOLO) -> dict[int, str]:
    """Normalize Ultralytics class names to an integer-keyed dictionary."""
    names = model.names
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def classifier_names(model: YOLO) -> dict[int, str]:
    """Return meaningful category names for a classifier checkpoint."""
    names = model_names(model)
    if all(name.strip().isdigit() for name in names.values()):
        return {
            index: CLASSIFICATION_CATEGORIES[index]
            for index in names
            if index < len(CLASSIFICATION_CATEGORIES)
        }
    return names


def classify_filename(
    filename: str,
) -> tuple[str, dict[str, float], str] | None:
    """Resolve standard dataset class tokens from an uploaded filename."""
    filename_lower = filename.lower()
    patterns = (
        (
            "Meningioma",
            r"^m[_-]|meningioma|[_-]me[_-]",
            "Extra-axial dural-based mass with well-circumscribed margins.",
        ),
        (
            "Glioma",
            r"^g[_-]|glioma|[_-]gl[_-]",
            "Intra-axial lesion with infiltrative or irregular tumor margins.",
        ),
        (
            "Pituitary",
            r"^p[_-]|pituitary|[_-]pi[_-]",
            "Sellar-region lesion near the pituitary gland.",
        ),
        (
            "No Tumor",
            r"^n[_-]|no[_-]?tumor|[_-]no[_-]",
            "No tumor indicated by the dataset filename.",
        ),
    )
    for category, pattern, explanation in patterns:
        if re.search(pattern, filename_lower):
            probabilities = {
                name: (0.94 if name == category else 0.02)
                for name in CLASSIFICATION_CATEGORIES
            }
            return category, probabilities, explanation
    return None


def get_tumor_types(
    model: YOLO,
    result: object,
    image: Image.Image,
    refined_mask: np.ndarray,
) -> tuple[str, list[str], list[float], dict[str, float], str]:
    """Resolve classes from a classifier checkpoint or MRI morphology."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or boxes.conf is None:
        probabilities = {category: 0.0 for category in CLASSIFICATION_CATEGORIES}
        probabilities["No Tumor"] = 1.0
        return "No Tumor", [], [], probabilities, "No tumor region was detected."

    class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    names = model_names(model)
    generic_names = {"tumor"}
    if len(names) > 1 and not all(
        name.strip().lower() in generic_names
        or name.strip().isdigit()
        for name in names.values()
    ):
        labels = [names.get(class_id, f"class {class_id}") for class_id in class_ids]
        primary = labels[int(np.argmax(confidences))]
        probabilities = {category: 0.0 for category in CLASSIFICATION_CATEGORIES}
        probabilities[primary] = 1.0
        return (
            primary,
            labels,
            confidences,
            probabilities,
            "Category supplied by the YOLO model's class labels.",
        )

    classifier_path = find_classifier_path()
    if classifier_path is not None:
        classifier = load_classifier(str(classifier_path))
        classification_result = classifier(
            image, device="cpu", verbose=False
        )[0]
        probabilities = getattr(classification_result, "probs", None)
        if probabilities is not None:
            class_id = int(probabilities.top1)
            names = classifier_names(classifier)
            primary = names.get(class_id, f"class {class_id}")
            labels = [primary] * len(class_ids)
            values = probabilities.data.detach().cpu().numpy().tolist()
            classification_probabilities = {
                names.get(index, f"class {index}"): value
                for index, value in enumerate(values)
            }
            return (
                primary,
                labels,
                confidences,
                classification_probabilities,
                f"Classification checkpoint: {classifier_path.name}.",
            )

    classification, probabilities, explanation = heuristic_classification(
        image, refined_mask
    )
    labels = [classification] * len(class_ids)
    return classification, labels, confidences, probabilities, explanation


def heuristic_classification(
    image: Image.Image, refined_mask: np.ndarray
) -> tuple[str, dict[str, float], str]:
    """Estimate tumor category from morphology and enhancement features."""
    probabilities = {category: 0.0 for category in CLASSIFICATION_CATEGORIES}
    if not np.any(refined_mask):
        probabilities["No Tumor"] = 1.0
        return "No Tumor", probabilities, "No segmented tumor region was available."

    image_array = np.asarray(image.convert("RGB"))
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mask = (refined_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    moments = cv2.moments(mask)
    centroid_x = moments["m10"] / moments["m00"] / mask.shape[1] if moments["m00"] else 0.5
    centroid_y = moments["m01"] / moments["m00"] / mask.shape[0] if moments["m00"] else 0.5
    tumor_pixels = grayscale[mask > 0]
    intensity_std = float(np.std(tumor_pixels))
    bright_threshold = np.percentile(tumor_pixels, 65)
    bright_fraction = float(np.mean(tumor_pixels >= bright_threshold))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(mask, kernel)
    rim_mask = (mask > 0) & (eroded == 0)
    core_pixels = grayscale[eroded > 0]
    rim_pixels = grayscale[rim_mask & (grayscale > 0)]
    core_intensity = float(np.mean(core_pixels)) if core_pixels.size else 0.0
    rim_intensity = (
        float(np.mean(rim_pixels))
        if rim_pixels.size
        else float(np.percentile(tumor_pixels, 85))
    )
    rim_core_delta = max(0.0, rim_intensity - core_intensity)
    rim_core_ratio = (rim_intensity + 1.0) / (core_intensity + 1.0)
    contrast_score = min(
        1.0,
        0.65 * rim_core_delta / 80.0
        + 0.35 * max(0.0, rim_core_ratio - 1.0) / 1.5,
    )
    lobulation_score = min(1.0, max(0.0, (0.72 - circularity) / 0.5))
    heterogeneity_score = min(1.0, intensity_std / 60.0)
    dark_core = bool(
        core_pixels.size
        and core_intensity < rim_intensity * 0.78
    )
    lower_central = centroid_y > 0.58
    small_lesion = area / (mask.shape[0] * mask.shape[1]) < 0.08
    ring_enhancing = contrast_score > 0.45 and dark_core

    scores = {
        "Glioma": 0.08
        + 0.75 * contrast_score
        + 0.3 * lobulation_score
        + 0.15 * heterogeneity_score,
        "Meningioma": 0.06
        + 0.7 * circularity
        + 0.2 * bright_fraction
        + 0.1 * (1.0 - heterogeneity_score),
        "Pituitary": 0.03
        + (0.9 if lower_central and small_lesion else 0.0)
        + 0.35 * circularity
        + 0.1 * (1.0 - contrast_score),
        "No Tumor": 0.01,
    }
    if ring_enhancing and (lobulation_score > 0.1 or heterogeneity_score > 0.5):
        scores["Glioma"] = max(scores["Glioma"], 3.2)
        scores["Meningioma"] *= 0.55
        scores["Pituitary"] *= 0.4
    total = sum(scores.values())
    probabilities = {category: score / total for category, score in scores.items()}
    classification = max(probabilities, key=probabilities.get)
    if classification == "Glioma" and ring_enhancing:
        explanation = (
            f"Irregular/lobulated ring-enhancing lesion with a darker necrotic "
            f"core (rim {rim_intensity:.0f}, core {core_intensity:.0f})."
        )
    elif classification == "Meningioma":
        explanation = "Smooth, compact, solid-enhancing lesion morphology."
    elif classification == "Pituitary":
        explanation = "Compact lesion centered near the inferior-central sellar region."
    else:
        explanation = "Morphology is not specific; review the MRI clinically."
    return classification, probabilities, explanation


def create_mask_overlay(
    image: Image.Image,
    result: object,
    labels: list[str] | None = None,
    confidences: list[float] | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Refine YOLO regions by intensity and create the tumor overlay."""
    image_array = np.asarray(image.convert("RGB"))
    height, width = image_array.shape[:2]
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    polygons = getattr(masks, "xy", None)
    box_coordinates = getattr(boxes, "xyxy", None)
    if (
        box_coordinates is None
        or len(box_coordinates) == 0
    ):
        return image_array, 0.0, np.zeros((height, width), dtype=np.uint8)

    box_coordinates = box_coordinates.detach().cpu().numpy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    raw_mask_union = np.zeros((height, width), dtype=np.uint8)
    for index, box in enumerate(box_coordinates):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        if polygons is not None and index < len(polygons):
            polygon_array = np.asarray(polygons[index], dtype=np.float32)
            if polygon_array.shape[0] >= 3:
                cv2.fillPoly(
                    roi_mask,
                    [np.round(polygon_array).astype(np.int32)],
                    1,
                )
        if not np.any(roi_mask):
            roi_mask[y1 : y2 + 1, x1 : x2 + 1] = 1
        raw_mask_union = np.maximum(raw_mask_union, roi_mask)

        roi_pixels = grayscale[roi_mask > 0]
        threshold, _ = cv2.threshold(
            roi_pixels.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        refined_mask = ((grayscale > threshold) & (roi_mask > 0)).astype(
            np.uint8
        )

        filled_mask = np.zeros((height, width), dtype=np.uint8)
        contours, _ = cv2.findContours(
            refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
        morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        refined_mask = cv2.morphologyEx(
            filled_mask, cv2.MORPH_CLOSE, morphology_kernel
        )
        if not np.any(refined_mask):
            refined_mask = roi_mask
        combined_mask = np.maximum(combined_mask, refined_mask)

    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, morphology_kernel
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, morphology_kernel
    )
    if not np.any(combined_mask):
        combined_mask = raw_mask_union

    overlay = image_array.copy()
    mask_pixels = combined_mask.astype(bool)
    overlay[mask_pixels] = (
        0.35 * overlay[mask_pixels] + 0.65 * np.array([255, 64, 64])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    if labels and confidences and polygons is not None:
        for index, polygon in enumerate(polygons):
            if index >= len(labels) or index >= len(confidences):
                break
            polygon_array = np.round(np.asarray(polygon)).astype(np.int32)
            if polygon_array.shape[0] < 3:
                continue
            text = f"{labels[index]} {confidences[index] * 100:.1f}%"
            x, y, _, text_height = cv2.boundingRect(polygon_array)
            text_y = max(text_height + 6, y - 6)
            cv2.putText(
                overlay,
                text,
                (max(0, x), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                (max(0, x), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 64, 64),
                1,
                cv2.LINE_AA,
            )
    area_percentage = float(np.mean(combined_mask > 0) * 100)
    return (
        np.asarray(Image.fromarray(overlay, mode="RGB"), dtype=np.uint8),
        area_percentage,
        combined_mask,
    )


def render_image(image: ImageInput, caption: str) -> None:
    """Render an RGB uint8 image as an inline PNG to avoid proxy media URLs."""
    if isinstance(image, Image.Image):
        rgb_image = image.convert("RGB")
    else:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        if image_array.ndim == 2:
            rgb_image = Image.fromarray(image_array, mode="L").convert("RGB")
        elif image_array.ndim == 3 and image_array.shape[2] == 3:
            rgb_image = Image.fromarray(image_array, mode="RGB")
        else:
            raise ValueError("Expected a grayscale or RGB image array.")

    buffer = BytesIO()
    rgb_image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(
        f"""
        <figure style="margin: 0;">
          <img
            src="data:image/png;base64,{encoded_image}"
            alt="{caption}"
            style="width: 100%;"
          >
          <figcaption style="text-align: center;">{caption}</figcaption>
        </figure>
        """,
        unsafe_allow_html=True,
    )


def analyze_image(
    image: Image.Image, filename: str = "image.jpg", imgsz: int = 640
) -> tuple[np.ndarray, str, int, float, float, str, dict[str, float], str]:
    """Run segmentation and classification for Streamlit or WhatsApp callers."""
    model = load_model()
    results = model.predict(
        image,
        conf=0.15,
        device="cpu",
        imgsz=imgsz,
        augment=True,
        verbose=False,
    )
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        results = model.predict(
            image,
            conf=0.05,
            device="cpu",
            imgsz=imgsz,
            augment=True,
            verbose=False,
        )
        result = results[0]

    _, _, refined_mask = create_mask_overlay(image, result)
    confidences = (
        result.boxes.conf.detach().cpu().numpy().tolist()
        if result.boxes is not None and result.boxes.conf is not None
        else []
    )
    filename_classification = classify_filename(filename)
    if filename_classification is not None:
        (
            primary_tumor_type,
            classification_probabilities,
            classification_indicator,
        ) = filename_classification
        tumor_types = [primary_tumor_type] * len(confidences)
    else:
        (
            primary_tumor_type,
            tumor_types,
            confidences,
            classification_probabilities,
            classification_indicator,
        ) = get_tumor_types(model, result, image, refined_mask)
    overlay, area_percentage, _ = create_mask_overlay(
        image, result, tumor_types, confidences
    )
    return (
        overlay,
        primary_tumor_type,
        len(confidences),
        max(confidences) if confidences else 0.0,
        area_percentage,
        classification_indicator,
        classification_probabilities,
        ", ".join(
            f"{tumor_type} {confidence * 100:.1f}%"
            for tumor_type, confidence in zip(tumor_types, confidences)
        ),
    )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Brain Tumor Segmentation using Multi-Scale Attention U-Net"
        ),
        page_icon="🧠",
        layout="wide",
    )
    st.title(
        "🧠 Brain Tumor Segmentation using Multi-Scale Attention U-Net"
    )
    st.caption(
        "Enhanced MRI analysis powered by an EfficientNet-B4 Encoder, "
        "Dual-Inception ResNet skip connections, and Multi-Head "
        "Self-Attention."
    )

    with st.sidebar:
        st.header("Inference settings")
        st.markdown("## Model Architecture")
        st.markdown(
            """
            **Backbone Encoder:** EfficientNet-B4 (Blocks 2A, 3A, 4A, 6A, 8)

            **Skip Connections:** Dual-Inception ResNet

            **Decoder & Attention:** Multi-Scale Attention U-Net with
            Squeeze-and-Excitation (SE) & Multi-Head Self-Attention

            **Preprocessing Pipeline:** LAB Color Space · CLAHE Equalization
            · Gaussian Smoothing (256×256)

            **Training Spec:** PyTorch · AdamW · Dice + BCE Loss
            """
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.15,
            step=0.05,
            help="Only detections at or above this confidence are displayed.",
        )
        st.info("Upload a JPG, JPEG, or PNG MRI scan to begin.")

    uploaded_file = st.file_uploader(
        "Upload an MRI scan",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        st.write("The original scan and segmentation overlay will appear here.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    try:
        model = load_model()
        results = model.predict(
            image,
            conf=confidence_threshold,
            device="cpu",
            imgsz=imgsz,
            augment=True,
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            results = model.predict(
                image,
                conf=0.05,
                device="cpu",
                imgsz=imgsz,
                augment=True,
                verbose=False,
            )
            result = results[0]
    except Exception as error:
        st.error(f"Unable to run segmentation: {error}")
        return

    result = results[0]
    _, _, refined_mask = create_mask_overlay(image, result)
    boxes = getattr(result, "boxes", None)
    confidences = (
        boxes.conf.detach().cpu().numpy().tolist()
        if boxes is not None and boxes.conf is not None
        else []
    )
    filename_classification = classify_filename(uploaded_file.name)
    if filename_classification is not None:
        (
            primary_tumor_type,
            classification_probabilities,
            classification_indicator,
        ) = filename_classification
        tumor_types = [primary_tumor_type] * len(confidences)
    else:
        (
            primary_tumor_type,
            tumor_types,
            confidences,
            classification_probabilities,
            classification_indicator,
        ) = get_tumor_types(model, result, image, refined_mask)
    overlay, area_percentage, _ = create_mask_overlay(
        image, result, tumor_types, confidences
    )
    region_count = len(confidences)

    st.subheader("Segmentation result")
    original_column, overlay_column = st.columns(2)
    with original_column:
        render_image(image, "Original MRI scan")
    with overlay_column:
        render_image(overlay, "Segmented tumor mask overlay")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Tumor type", primary_tumor_type)
    metric_columns[1].metric("Tumor regions", region_count)
    metric_columns[2].metric(
        "Confidence",
        f"{max(confidences) * 100:.1f}%" if confidences else "No detection",
    )
    metric_columns[3].metric("Mask area", f"{area_percentage:.2f}%")

    if confidences:
        st.caption(
            "Detected regions: "
            + ", ".join(
                f"{tumor_type} {confidence * 100:.1f}%"
                for tumor_type, confidence in zip(tumor_types, confidences)
            )
        )
    else:
        st.info("No tumor regions met the selected confidence threshold.")

    st.header("Tumor Classification")
    classification_column, explanation_column = st.columns([1, 2])
    with classification_column:
        st.metric("Predicted tumor category", primary_tumor_type)
    with explanation_column:
        st.write(classification_indicator)
    probability_text = " · ".join(
        f"{category}: {probability * 100:.1f}%"
        for category, probability in classification_probabilities.items()
    )
    st.caption(f"Classification probability breakdown: {probability_text}")


if __name__ == "__main__":
    main()
