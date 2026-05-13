from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency guard
    fitz = None


PHOTO_MAX_BYTES = 200 * 1024
SIGNATURE_MAX_BYTES = 100 * 1024
PDF_MAX_BYTES = 200 * 1024

PHOTO_MIN_WIDTH = 250
PHOTO_MIN_HEIGHT = 300
SIGNATURE_MIN_WIDTH = 200

PHOTO_RATIO_RANGES = [(0.65, 0.85), (0.95, 1.05)]
SIGNATURE_RATIO_RANGE = (1.5, 4.5)

PHOTO_TARGET_RATIO = 0.78
PHOTO_TARGET_SIZE = (600, 750)
PDF_DEFAULT_DPI = 140
PDF_DEFAULT_QUALITY = 75



@dataclass
class ValidationResult:
    ok: bool
    messages: list[str]


def pil_image_from_upload(uploaded_file) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGBA")


def image_from_data_url(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data)).convert("RGBA")




def image_to_bytes(image: Image.Image, format_name: str, **save_kwargs) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format_name, **save_kwargs)
    return buffer.getvalue()


def reencode_to_limit(image: Image.Image, target_bytes: int, preferred_formats: tuple[str, ...] = ("JPEG", "PNG")) -> tuple[bytes, str]:
    best_bytes = b""
    best_format = preferred_formats[0]

    for format_name in preferred_formats:
        if format_name == "JPEG":
            rgb_image = image.convert("RGB")
            for quality in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
                data = image_to_bytes(rgb_image, "JPEG", quality=quality, optimize=True, progressive=True)
                if not best_bytes or len(data) < len(best_bytes):
                    best_bytes = data
                    best_format = "JPEG"
                if len(data) <= target_bytes:
                    return data, "JPEG"
        elif format_name == "PNG":
            data = image_to_bytes(image, "PNG", optimize=True)
            if not best_bytes or len(data) < len(best_bytes):
                best_bytes = data
                best_format = "PNG"
            if len(data) <= target_bytes:
                return data, "PNG"

    return best_bytes, best_format


def get_laplacian_variance(image: Image.Image) -> float:
    if cv2 is None:
        grayscale = np.array(ImageOps.grayscale(image))
        laplacian = np.abs(np.diff(grayscale.astype(np.float32), axis=0)).mean() + np.abs(np.diff(grayscale.astype(np.float32), axis=1)).mean()
        return float(laplacian)

    grayscale = np.array(ImageOps.grayscale(image))
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    if cv2 is None:
        return []

    image_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [tuple(map(int, face)) for face in faces]


def crop_to_aspect(image: Image.Image, target_ratio: float, focus_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    width, height = image.size
    current_ratio = width / height

    if math.isclose(current_ratio, target_ratio, rel_tol=0.01):
        return image

    if focus_box is not None:
        x, y, w, h = focus_box
        focus_center_x = x + w / 2
        focus_center_y = y + h / 2
    else:
        focus_center_x = width / 2
        focus_center_y = height / 2

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = int(max(0, min(width - new_width, focus_center_x - new_width / 2)))
        crop_box = (left, 0, left + new_width, height)
    else:
        new_height = int(width / target_ratio)
        top = int(max(0, min(height - new_height, focus_center_y - new_height / 2)))
        crop_box = (0, top, width, top + new_height)

    return image.crop(crop_box)


def resize_for_output(image: Image.Image, min_width: int, min_height: int, target_size: tuple[int, int] | None = None) -> Image.Image:
    width, height = image.size
    if width < min_width or height < min_height:
        scale = max(min_width / width, min_height / height)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image


def estimate_background_uniformity(image: Image.Image) -> float:
    grayscale = ImageOps.grayscale(image)
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges, dtype=np.float32)

    border = int(min(image.size) * 0.12)
    border = max(border, 12)
    border_mask = np.zeros(edge_array.shape, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    border_edges = edge_array[border_mask]
    return float(border_edges.mean() if border_edges.size else 0.0)


def estimate_color_richness(image: Image.Image) -> float:
    rgb = np.array(image.convert("RGB"))
    sample = rgb[:: max(rgb.shape[0] // 100, 1), :: max(rgb.shape[1] // 100, 1)]
    if sample.size == 0:
        return 0.0
    unique = np.unique(sample.reshape(-1, 3), axis=0)
    return float(len(unique))


def validate_photo(image: Image.Image) -> ValidationResult:
    messages: list[str] = []
    width, height = image.size
    ratio = width / height
    faces = detect_faces(image)

    if width < PHOTO_MIN_WIDTH or height < PHOTO_MIN_HEIGHT:
        messages.append(f"Photo resolution is too small. Minimum is {PHOTO_MIN_WIDTH} x {PHOTO_MIN_HEIGHT}px.")

    if not any(low <= ratio <= high for low, high in PHOTO_RATIO_RANGES):
        messages.append("Photo aspect ratio should match standard passport sizing (0.65-0.85 or 0.95-1.05).")

    if len(faces) != 1:
        messages.append("Exactly one face must be detected in the photo.")
    else:
        face_x, face_y, face_w, face_h = faces[0]
        face_area_ratio = (face_w * face_h) / (width * height)
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2

        if not (0.04 <= face_area_ratio <= 0.65):
            messages.append("The face occupies an unusual amount of the image area; adjust the framing.")

        if abs(face_center_x - width / 2) > width * 0.12 or abs(face_center_y - height / 2) > height * 0.12:
            messages.append("Face should be horizontally and vertically centered.")

    blur_score = get_laplacian_variance(image)
    if blur_score < 80:
        messages.append("Photo appears blurry or out of focus.")

    background_score = estimate_background_uniformity(image)
    if background_score > 18:
        messages.append("Background looks busy or textured; a plain light background works best.")

    richness = estimate_color_richness(image)
    if richness < 400:
        messages.append("Photo looks too flat or low-detail for a natural portrait; check for cartoons, avatars, or heavy filters.")

    return ValidationResult(ok=not messages, messages=messages)


def process_photo(image: Image.Image, target_ratio: float, target_size: tuple[int, int]) -> tuple[bytes, str, ValidationResult, Image.Image]:
    validation = validate_photo(image)
    faces = detect_faces(image)
    focus_box = faces[0] if faces else None
    working = crop_to_aspect(image.convert("RGB"), target_ratio, focus_box)
    working = resize_for_output(working, PHOTO_MIN_WIDTH, PHOTO_MIN_HEIGHT, target_size)
    data, file_format = reencode_to_limit(working, PHOTO_MAX_BYTES, ("JPEG", "PNG"))
    return data, file_format.lower(), validation, working


def validate_signature(image: Image.Image) -> ValidationResult:
    messages: list[str] = []
    width, height = image.size
    ratio = width / height

    if width < SIGNATURE_MIN_WIDTH:
        messages.append(f"Signature width must be at least {SIGNATURE_MIN_WIDTH}px.")

    if not (SIGNATURE_RATIO_RANGE[0] <= ratio <= SIGNATURE_RATIO_RANGE[1]):
        messages.append("Signature aspect ratio should be rectangular and between 1.5 and 4.5.")

    faces = detect_faces(image)
    if len(faces) > 0:
        messages.append("A face was detected in the signature file; upload only the signature.")

    grayscale = ImageOps.grayscale(image.convert("RGB"))
    inverted = ImageOps.invert(grayscale)
    threshold = inverted.point(lambda p: 255 if p > 40 else 0)
    ink_pixels = np.array(threshold) > 0
    ink_ratio = float(ink_pixels.mean())

    if ink_ratio < 0.003:
        messages.append("Signature appears blank or nearly blank.")

    connected_components = 0
    if cv2 is not None:
        _component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(np.array(threshold), connectivity=8)
        connected_components = int(sum(1 for area in stats[1:, cv2.CC_STAT_AREA] if area >= 10))
    else:
        connected_components = int(min(75, max(1, ink_ratio * 1000)))

    if connected_components > 50:
        messages.append("Signature contains too many disconnected strokes or excessive noise.")

    border = int(min(width, height) * 0.12)
    border = max(border, 10)
    border_array = np.array(ImageOps.grayscale(image))
    border_pixels = np.concatenate([
        border_array[:border, :].ravel(),
        border_array[-border:, :].ravel(),
        border_array[:, :border].ravel(),
        border_array[:, -border:].ravel(),
    ])
    if border_pixels.size and border_pixels.mean() < 220:
        messages.append("Signature background should be white paper or transparent.")

    return ValidationResult(ok=not messages, messages=messages)


def process_signature(image: Image.Image) -> tuple[bytes, str, ValidationResult, Image.Image]:
    validation = validate_signature(image)
    working = image.convert("RGBA")

    alpha = working.getchannel("A")
    if alpha.getbbox() is not None:
        background = Image.new("RGBA", working.size, (255, 255, 255, 255))
        working = Image.alpha_composite(background, working).convert("RGB")
    else:
        working = working.convert("RGB")

    grayscale = ImageOps.grayscale(working)
    inverted = ImageOps.invert(grayscale)
    bbox = inverted.point(lambda p: 255 if p > 35 else 0).getbbox()
    if bbox:
        working = working.crop(bbox)

    if working.width < SIGNATURE_MIN_WIDTH:
        scale = SIGNATURE_MIN_WIDTH / working.width
        working = working.resize((SIGNATURE_MIN_WIDTH, max(1, int(working.height * scale))), Image.Resampling.LANCZOS)

    data, file_format = reencode_to_limit(working, SIGNATURE_MAX_BYTES, ("PNG", "JPEG"))
    return data, file_format.lower(), validation, working


def compress_pdf(uploaded_file, target_dpi: int = PDF_DEFAULT_DPI, jpeg_quality: int = PDF_DEFAULT_QUALITY) -> tuple[bytes | None, str]:
    if fitz is None:
        return None, "PyMuPDF is not installed"

    source_bytes = uploaded_file.getvalue()
    dpi_candidates = [target_dpi, max(90, target_dpi - 20), max(90, target_dpi - 40), max(90, target_dpi - 60)]
    quality_candidates = [jpeg_quality, max(50, jpeg_quality - 10), max(40, jpeg_quality - 20)]
    best_pdf_bytes: bytes | None = None
    best_label = "compression failed"

    for dpi in dpi_candidates:
        for quality in quality_candidates:
            document = fitz.open(stream=source_bytes, filetype="pdf")
            output = fitz.open()

            for page in document:
                rect = page.rect
                new_page = output.new_page(width=rect.width, height=rect.height)
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                image_bytes = image_to_bytes(image, "JPEG", quality=quality, optimize=True, progressive=True)
                new_page.insert_image(rect, stream=image_bytes)

            pdf_bytes = output.write(deflate=True, garbage=4, clean=True)
            output.close()
            document.close()

            if best_pdf_bytes is None or len(pdf_bytes) < len(best_pdf_bytes):
                best_pdf_bytes = pdf_bytes
                best_label = f"{dpi} dpi / quality {quality}"

            if len(pdf_bytes) <= PDF_MAX_BYTES:
                return pdf_bytes, f"compressed to {filesize_label(len(pdf_bytes))} at {dpi} dpi / quality {quality}"

    if best_pdf_bytes is None:
        return None, "Unable to compress PDF"

    return best_pdf_bytes, f"best effort {best_label} -> {filesize_label(len(best_pdf_bytes))}"


def filesize_label(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def safe_filename_part(value: str) -> str:
    cleaned = "".join(character for character in value.strip().lower() if character.isalnum())
    return cleaned or "student"


def nudge_image_rotation(prefix: str, degrees: int) -> None:
    current_value = int(st.session_state.get(f"{prefix}_rotation", 0))
    st.session_state[f"{prefix}_rotation"] = current_value + degrees


def apply_image_edits(
    image: Image.Image,
    brightness: float,
    rotation: int,
) -> Image.Image:
    working = ImageOps.exif_transpose(image).convert("RGBA")

    if rotation:
        working = working.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    if brightness != 1.0:
        working = ImageEnhance.Brightness(working).enhance(brightness)

    return working


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    rgba_image = image.convert("RGBA")
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba_image).convert("RGB")


def jpeg_bytes_under_limit(image: Image.Image, target_bytes: int) -> bytes:
    working = flatten_to_rgb(image)
    best_bytes = b""

    for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
        candidate = working
        if scale < 1.0:
            candidate = working.resize(
                (max(1, int(working.width * scale)), max(1, int(working.height * scale))),
                Image.Resampling.LANCZOS,
            )

        for quality in (90, 80, 70, 60, 50):
            data = image_to_bytes(candidate, "JPEG", quality=quality, optimize=True, progressive=True)
            if not best_bytes or len(data) < len(best_bytes):
                best_bytes = data
            if len(data) <= target_bytes:
                return data

    return best_bytes


def pdf_bytes_from_image(image: Image.Image, quality: int) -> bytes:
    rgb_image = flatten_to_rgb(image)

    if fitz is None:
        buffer = BytesIO()
        rgb_image.save(buffer, format="PDF", resolution=72.0)
        return buffer.getvalue()

    jpeg_data = image_to_bytes(rgb_image, "JPEG", quality=quality, optimize=True, progressive=True)
    document = fitz.open()
    page = document.new_page(width=rgb_image.width, height=rgb_image.height)
    page.insert_image(page.rect, stream=jpeg_data)
    pdf_bytes = document.write(deflate=True, garbage=4, clean=True)
    document.close()
    return pdf_bytes


def image_to_pdf_under_limit(image: Image.Image, target_bytes: int) -> bytes:
    working = flatten_to_rgb(ImageOps.exif_transpose(image))
    best_bytes = b""

    for scale in (1.0, 0.75, 0.5, 0.35):
        candidate = working
        if scale < 1.0:
            candidate = working.resize(
                (max(1, int(working.width * scale)), max(1, int(working.height * scale))),
                Image.Resampling.LANCZOS,
            )

        for quality in (85, 70, 55, 45):
            pdf_bytes = pdf_bytes_from_image(candidate, quality)
            if not best_bytes or len(pdf_bytes) < len(best_bytes):
                best_bytes = pdf_bytes
            if len(pdf_bytes) <= target_bytes:
                return pdf_bytes

    return best_bytes


def prepare_document_output(uploaded_file) -> tuple[bytes | None, str]:
    file_name = (getattr(uploaded_file, "name", "") or "").lower()
    file_type = (getattr(uploaded_file, "type", "") or "").lower()
    is_pdf = file_name.endswith(".pdf") or file_type == "application/pdf"

    if is_pdf:
        pdf_bytes, status = compress_pdf(uploaded_file)
        if pdf_bytes is None:
            return None, status
        return pdf_bytes, status

    try:
        image = pil_image_from_upload(uploaded_file)
    except Exception:
        return None, "Unsupported file type"

    return image_to_pdf_under_limit(image, PDF_MAX_BYTES), "Converted image to PDF"


def render_editable_image_section(
    uploaded_file,
    *,
    key_prefix: str,
    output_stub: str,
    validation_fn,
    min_width: int,
    min_height: int,
) -> None:
    original_image = pil_image_from_upload(uploaded_file)

    left_col, middle_col, right_col = st.columns([1.5, 1.2, 1.5], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown("**Original Image**")
            st.image(original_image, width=120)
            st.caption(f"{original_image.width} × {original_image.height} px")

    with middle_col:
        with st.container(border=True):
            st.markdown("**Brightness**")
            brightness = st.slider(
                "Adjust brightness",
                0.5,
                2.0,
                float(st.session_state.get(f"{key_prefix}_brightness", 1.0)),
                0.05,
                key=f"{key_prefix}_brightness",
                label_visibility="collapsed",
            )

        st.markdown("")
        with st.container(border=True):
            st.markdown("**Rotate**")
            rotate_left, rotate_right = st.columns(2)
            with rotate_left:
                if st.button("↺", key=f"{key_prefix}_rot_left", use_container_width=True, help="Rotate left 90°"):
                    nudge_image_rotation(key_prefix, -90)
            with rotate_right:
                if st.button("↻", key=f"{key_prefix}_rot_right", use_container_width=True, help="Rotate right 90°"):
                    nudge_image_rotation(key_prefix, 90)

    with right_col:
        with st.container(border=True):
            st.markdown("**Preview**")
            rotation = int(st.session_state.get(f"{key_prefix}_rotation", 0))
            edited_image = apply_image_edits(original_image, brightness, rotation)
            output_image = edited_image
            if min_width or min_height:
                output_image = resize_for_output(edited_image, min_width, min_height)

            validation = validation_fn(output_image)
            image_bytes = jpeg_bytes_under_limit(output_image, PHOTO_MAX_BYTES if output_stub == "photo" else SIGNATURE_MAX_BYTES)

            st.image(output_image, width=120)
            st.caption(f"{output_image.width} × {output_image.height} px | {filesize_label(len(image_bytes))}")

    st.markdown("")
    if validation.ok:
        st.success("Image is ready to download")
    else:
        st.warning("Image needs review")
        for message in validation.messages:
            st.caption(f"• {message}")

    st.download_button(
        f"Download {output_stub.title()}.jpg",
        data=image_bytes,
        file_name=f"{filename_prefix}_{output_stub}.jpg",
        mime="image/jpeg",
        key=f"download_{key_prefix}",
        use_container_width=True,
    )


def render_document_section(upload_label: str, *, key_prefix: str, output_stub: str) -> None:
    uploaded_document = st.file_uploader(upload_label, type=["pdf", "jpg", "jpeg", "png"], key=key_prefix)
    if not uploaded_document:
        return

    st.caption(f"Input: {filesize_label(uploaded_document.size)}")
    pdf_bytes, status = prepare_document_output(uploaded_document)
    if pdf_bytes is None:
        st.error(f"Document conversion is unavailable: {status}")
        return

    st.caption(f"Output: {filesize_label(len(pdf_bytes))}")
    st.download_button(
        f"Download {output_stub} PDF",
        data=pdf_bytes,
        file_name=f"{filename_prefix}_{output_stub}.pdf",
        mime="application/pdf",
        key=f"download_{key_prefix}",
    )




# Note: UI simplified per user request — rules, clipboard paste, and copy-to-clipboard removed.


st.set_page_config(page_title="BCWCC Admission Helper", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.4rem;
    }
    .hero {
        display: flex !important;
        align-items: center !important;
        padding: 10px 20px !important;
        border-radius: 14px;
        background: linear-gradient(100deg, #1f553f 0%, #d9b720 75%, #f2c200 100%);
        color: #ffffff !important;
        box-shadow: 0 20px 40px rgba(11, 47, 36, 0.14);
        margin-bottom: 0.8rem;
        min-height: 64px !important;
    }
    .hero-title {
        margin: 2rem !important;
        font-size: 72px !important;
        font-weight: 600 !important;
        padding: 0 !important;
        line-height: 1 !important;
        color: #ffffff !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.45) !important;
        white-space: nowrap;
    }
    .stApp {
        background: linear-gradient(180deg, #f5f7f3 0%, #ffffff 100%);
    }
    .stButton > button {
        background: #f2c200 !important;
        color: #0b2f24 !important;
        border: 1px solid #d4a900 !important;
        font-weight: 700 !important;
        border-radius: 0.7rem !important;
    }
    .stDownloadButton > button {
        background: #0f8b3a !important;
        color: #ffffff !important;
        border: 1px solid transparent !important;
        font-weight: 700 !important;
        border-radius: 0.7rem !important;
        padding: 0.6rem 1rem !important;
    }
    /* Ensure nested text inside download buttons is white */
    .stDownloadButton button, .stDownloadButton > button, .stDownloadButton span, .stDownloadButton > button * {
        color: #ffffff !important;
    }
    /* Constrain file uploader to a compact box to avoid large gaps */
    .stFileUploader, .stFileUploader > div {
        max-width: 360px !important;
        width: 360px !important;
        height: auto !important;
        min-height: 120px !important;
        max-height: 220px !important;
        display: block;
        margin-bottom: 8px !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    .stFileUploader .css-1r6slb0 { /* dropzone inner (best effort) */
        height: auto !important;
        min-height: 120px !important;
    }
    /* Make small image previews sit closer to uploader */
    .stImage img {
        margin-top: 6px !important;
        margin-left: 0 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    /* Make text larger and darker for readability (less aggressive specificity) */
    body, label, p, span, h2, h3, h4, .stMarkdown, .stCaption, .stText {
        color: #072014;
        font-size: 15px;
        font-weight: 600;
    }
    /* Ensure the hero heading remains bright and visible */
    .hero, .hero .hero-title, .hero .hero-title * { color: #ffffff !important; text-shadow: 0 1px 3px rgba(0,0,0,0.45) !important; }
    h2 { font-size: 20px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">BCWCC Admission Helper</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# rules removed per user request

student_first_name = st.text_input("Student first name", placeholder="Enter first name for file names")
filename_prefix = safe_filename_part(student_first_name) if student_first_name else "student"

tab_photo, tab_signature, tab_pdf = st.tabs(["Photo", "Signature", "Marks Card PDF"])

with tab_photo:
    uploaded_photo = st.file_uploader("Upload student photo", type=["jpg", "jpeg", "png"], key="photo")
    if uploaded_photo:
        render_editable_image_section(
            uploaded_photo,
            key_prefix="photo",
            output_stub="photo",
            validation_fn=validate_photo,
            min_width=PHOTO_MIN_WIDTH,
            min_height=PHOTO_MIN_HEIGHT,
        )

with tab_signature:
    uploaded_signature = st.file_uploader("Upload signature image", type=["jpg", "jpeg", "png"], key="signature")
    if uploaded_signature:
        render_editable_image_section(
            uploaded_signature,
            key_prefix="signature",
            output_stub="signature",
            validation_fn=validate_signature,
            min_width=SIGNATURE_MIN_WIDTH,
            min_height=1,
        )

with tab_pdf:
    pdf_choice_tabs = st.tabs(["10th marks card", "12th marks card", "Caste certificate", "Income certificate"])

    with pdf_choice_tabs[0]:
        render_document_section("Upload 10th marks card PDF or image", key_prefix="pdf10", output_stub="10th_marks_card")

    with pdf_choice_tabs[1]:
        render_document_section("Upload 12th marks card PDF or image", key_prefix="pdf12", output_stub="12th_marks_card")

    with pdf_choice_tabs[2]:
        render_document_section("Upload caste certificate PDF or image", key_prefix="caste_cert", output_stub="caste_certificate")

    with pdf_choice_tabs[3]:
        render_document_section("Upload income certificate PDF or image", key_prefix="income_cert", output_stub="income_certificate")