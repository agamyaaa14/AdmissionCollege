from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import declare_component
from PIL import Image, ImageFilter, ImageOps

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

COMPONENT_DIR = Path(__file__).parent / "components" / "clipboard_image"
clipboard_image_component = declare_component("clipboard_image", path=str(COMPONENT_DIR))


@dataclass
class ValidationResult:
    ok: bool
    messages: list[str]


def pil_image_from_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGBA")


def image_from_data_url(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data)).convert("RGBA")


def parse_pasted_image(payload: Any) -> tuple[Image.Image | None, str | None]:
    if not payload or not isinstance(payload, dict):
        return None, None
    data_url = payload.get("dataUrl")
    if not data_url:
        return None, None
    image = image_from_data_url(data_url)
    file_name = payload.get("fileName") or "pasted_image.png"
    return image, file_name


def render_clipboard_image_input(key: str) -> tuple[Image.Image | None, str | None]:
    st.caption("Or paste an image from the clipboard below.")
    try:
        payload = clipboard_image_component(key=key)
    except Exception:
        # Fallback when component frontend can't be served (network/proxy/permission issues)
        st.info(
            "Paste support is unavailable in this environment.\nPlease paste the image into an image editor (Paint), save it, then upload using the file picker."
        )
        return None, None

    return parse_pasted_image(payload)


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


def render_copy_button(file_bytes: bytes, mime_type: str, button_label: str, button_id: str) -> None:
    payload = base64.b64encode(file_bytes).decode("ascii")
    html = f"""
    <button id="{button_id}" style="
        padding: 0.65rem 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(15, 23, 42, 0.15);
        background: #0f172a;
        color: white;
        cursor: pointer;
        font-weight: 600;
        width: 100%;
    ">{button_label}</button>
    <script>
    const button = document.getElementById("{button_id}");
    button.addEventListener("click", async () => {{
        try {{
            const response = await fetch("data:{mime_type};base64,{payload}");
            const blob = await response.blob();
            if (navigator.clipboard && window.ClipboardItem) {{
                await navigator.clipboard.write([new ClipboardItem({{ "{mime_type}": blob }})]);
                button.innerText = "Copied";
                setTimeout(() => button.innerText = "{button_label}", 1400);
            }} else {{
                button.innerText = "Clipboard not supported";
                setTimeout(() => button.innerText = "{button_label}", 1800);
            }}
        }} catch (error) {{
            button.innerText = "Copy failed";
            setTimeout(() => button.innerText = "{button_label}", 1800);
        }}
    }});
    </script>
    """
    components.html(html, height=70)


def display_rules() -> None:
    with st.expander("Rules", expanded=False):
        st.markdown("### Photo Rules")
        st.markdown(
            "- JPG, JPEG, PNG\n- Maximum 200 KB\n- Minimum 250 x 300 px\n- Passport-style ratio: 0.65-0.85 or 0.95-1.05\n- Exactly one centered face\n- Plain light background\n- Rejects blurry, busy, cartoon-like, or avatar-style images"
        )
        st.markdown("### Signature Rules")
        st.markdown(
            "- JPG, JPEG, PNG\n- Maximum 100 KB\n- Width at least 200 px\n- Ratio between 1.5 and 4.5\n- White or transparent background\n- No face in the image\n- Rejects blank or noisy signatures"
        )
        st.markdown("### Marks Card PDF")
        st.markdown("- Optimizes scanned PDFs to help get them under 200 KB\n- Uses raster compression, so document clarity depends on the chosen DPI")


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
        background: #124734 !important;
        color: #ffffff !important;
        border: 1px solid #0b2f24 !important;
        font-weight: 700 !important;
        border-radius: 0.7rem !important;
    }
    /* Constrain file uploader to a centered square */
    .stFileUploader, .stFileUploader > div {
        max-width: 360px !important;
        width: 360px !important;
        height: 360px !important;
        aspect-ratio: 1 / 1;
        display: block;
        margin-left: 0 !important;
        margin-right: 0 !important;
        float: left !important;
    }
    .stFileUploader .css-1r6slb0 { /* dropzone inner (best effort) */
        height: 100% !important;
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

display_rules()

student_first_name = st.text_input("Student first name", placeholder="Enter first name for file names")
filename_prefix = safe_filename_part(student_first_name) if student_first_name else "student"

tab_photo, tab_signature, tab_pdf = st.tabs(["Photo", "Signature", "Marks Card PDF"])

with tab_photo:
    uploaded_photo = st.file_uploader("Upload student photo", type=["jpg", "jpeg", "png"], key="photo")
    pasted_photo_image, pasted_photo_name = render_clipboard_image_input("photo-paste")
    photo_image = pil_image_from_upload(uploaded_photo) if uploaded_photo else pasted_photo_image

    if photo_image is not None:
        st.caption(f"{photo_image.width} x {photo_image.height}px")
        photo_bytes, photo_fmt, photo_validation, processed_photo = process_photo(photo_image, PHOTO_TARGET_RATIO, PHOTO_TARGET_SIZE)

        if photo_validation.ok:
            st.success("Ready")
        else:
            st.warning("Needs review")
            for message in photo_validation.messages:
                st.write(f"- {message}")

        st.image(processed_photo, caption="Processed photo preview", use_container_width=True)
        st.caption(f"Processed size: {filesize_label(len(photo_bytes))} | Output format: {photo_fmt.upper()}")
        render_copy_button(photo_bytes, "image/jpeg" if photo_fmt == "jpeg" else "image/png", "Copy processed photo", "copy-photo")
        st.download_button(
            "Download processed photo",
            data=photo_bytes,
            file_name=f"{filename_prefix}_photo.{photo_fmt}",
            mime="image/jpeg" if photo_fmt == "jpeg" else "image/png",
            key="download_photo",
        )

with tab_signature:
    uploaded_signature = st.file_uploader("Upload signature image", type=["jpg", "jpeg", "png"], key="signature")
    pasted_signature_image, pasted_signature_name = render_clipboard_image_input("signature-paste")
    signature_image = pil_image_from_upload(uploaded_signature) if uploaded_signature else pasted_signature_image

    if signature_image is not None:
        st.caption(f"{signature_image.width} x {signature_image.height}px")
        signature_bytes, signature_fmt, signature_validation, processed_signature = process_signature(signature_image)

        if signature_validation.ok:
            st.success("Ready")
        else:
            st.warning("Needs review")
            for message in signature_validation.messages:
                st.write(f"- {message}")

        st.image(processed_signature, caption="Processed signature preview", use_container_width=True)
        st.caption(f"Processed size: {filesize_label(len(signature_bytes))} | Output format: {signature_fmt.upper()}")
        render_copy_button(signature_bytes, "image/jpeg" if signature_fmt == "jpeg" else "image/png", "Copy processed signature", "copy-signature")
        st.download_button(
            "Download processed signature",
            data=signature_bytes,
            file_name=f"{filename_prefix}_signature.{signature_fmt}",
            mime="image/jpeg" if signature_fmt == "jpeg" else "image/png",
            key="download_signature",
        )

with tab_pdf:
    uploaded_pdf = st.file_uploader("Upload marks card PDF", type=["pdf"], key="pdf")
    if uploaded_pdf:
        st.caption(f"{filesize_label(uploaded_pdf.size)}")
        pdf_bytes, pdf_status = compress_pdf(uploaded_pdf)
        if pdf_bytes is None:
            st.error(f"PDF compression is unavailable: {pdf_status}")
        else:
            st.success("Ready")
            st.caption(f"Compressed size: {filesize_label(len(pdf_bytes))}")
            render_copy_button(pdf_bytes, "application/pdf", "Copy compressed PDF", "copy-pdf")
            st.download_button(
                "Download compressed PDF",
                data=pdf_bytes,
                file_name=f"{filename_prefix}_marks_card.pdf",
                mime="application/pdf",
                key="download_pdf",
            )