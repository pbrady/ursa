import base64
import io
import logging
import mimetypes
from pathlib import Path

import pymupdf
from langchain.tools import ToolRuntime
from langchain_core.messages.content import (
    ImageContentBlock,
    TextContentBlock,
    create_image_block,
)
from langchain_core.tools import tool
from PIL import Image, UnidentifiedImageError

from ursa.agents.base import AgentContext

logger = logging.getLogger(__name__)

# Pillow format names that every vision endpoint URSA talks to accepts as-is
# (OpenAI lists png, jpeg, gif and webp; Anthropic and Gemini accept the same
# four). Anything else Pillow can decode (ppm/pgm/pbm, bmp, tiff, ico, ...)
# is re-encoded as PNG: OpenAI answers other bytes with a 400 "The image data
# you provided does not represent a valid image", and an agent that is not
# told why keeps resending them.
PASS_THROUGH_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# Longest side sent to the model. OpenAI scales high-detail images to fit in
# 2048x2048 and Anthropic refuses images over 8000 px, so a larger image only
# costs upload bytes; it is downscaled (keeping its aspect ratio) instead.
MAX_IMAGE_SIDE_PX = 2048

# Largest canvas decoded at all, checked from the header before any pixel is
# read, so a small file that declares an enormous canvas is refused rather
# than expanded in memory.
MAX_DECODE_PIXELS = 100_000_000

# Modes PNG stores as they are. Others (CMYK, YCbCr, LAB, HSV, float and
# 32-bit integer) are converted first.
_PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}
_HIGH_DEPTH_GRAY_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


class ImageReadError(ValueError):
    """The file cannot be sent to the model as an image.

    ``read_image_tool`` returns the message to the model as text, so it
    should say what to do instead (for example, convert the file to PNG).
    """


@tool
def read_image_tool(
    image_path: str, runtime: ToolRuntime[AgentContext]
) -> list[TextContentBlock | ImageContentBlock]:
    """Read an image from disk to ingest into the workflow"""
    workspace = runtime.context.workspace
    resolved_path = workspace.joinpath(image_path)

    try:
        display_path = str(resolved_path.relative_to(workspace))
    except ValueError:
        # Not under the workspace (e.g. an absolute path); label it the way
        # the caller referenced it.
        display_path = image_path

    try:
        image_block = image_block_from_file(resolved_path)
    except ImageReadError as e:
        # A file the model cannot be shown is the model's to deal with, not
        # a reason to fail the run or to send bytes the provider rejects.
        logger.warning(
            "Image read refused: %s",
            e,
            extra={"image_path": str(resolved_path)},
        )
        return [
            {
                "type": "text",
                "text": f"Could not read image file {display_path}: {e}",
            }
        ]
    except Exception as e:
        logger.exception(
            "Image read failed",
            exc_info=e,
            extra={"image_path": str(resolved_path)},
        )
        raise

    # The text block surfaces the filename to the model. It must stay a
    # separate text block: ImageContentBlock.file_id is reserved for
    # provider-side file-store references (OpenAI file IDs, Google File API
    # URIs), not local paths, and providers may give it priority over the
    # base64 payload.
    text_block: TextContentBlock = {
        "type": "text",
        "text": f"Image file: {display_path}",
    }
    return [text_block, image_block]


def image_block_from_file(
    filename: Path,
    max_size_mb: float = 20,
) -> ImageContentBlock:
    """Build a base64 image block the model API will accept.

    PNG, JPEG, still GIF and WebP no larger than ``MAX_IMAGE_SIDE_PX`` are
    sent unchanged. SVG is rendered to PNG. Any other raster format Pillow
    can decode, an animated GIF (first frame) and an image larger than
    ``MAX_IMAGE_SIDE_PX`` are re-encoded as PNG. Raises ``ImageReadError``
    for a missing, oversized or undecodable file.
    """
    max_bytes = max_size_mb * 1024 * 1024
    try:
        file_size = filename.stat().st_size
    except FileNotFoundError:
        raise ImageReadError("file not found") from None
    except OSError as e:
        raise ImageReadError(f"cannot read file: {e.strerror or e}") from None
    if file_size > max_bytes:
        raise ImageReadError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB "
            f"(max: {max_size_mb:0.1f}MB)"
        )

    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type == "image/svg+xml":
        with pymupdf.open(filename) as document:
            page = document[0]
            pixmap = page.get_pixmap()
            image_bytes = pixmap.tobytes("png")
    else:
        try:
            image_bytes = filename.read_bytes()
        except OSError as e:
            # e.g. a directory, or a file the agent may not read
            raise ImageReadError(
                f"cannot read file: {e.strerror or e}"
            ) from None

    image_bytes, mime_type = _normalize_image(image_bytes, filename, mime_type)
    if len(image_bytes) > max_bytes:
        raise ImageReadError(
            f"Encoded image too large: {len(image_bytes) / 1024 / 1024:.2f}MB "
            f"(max: {max_size_mb:0.1f}MB)"
        )

    data = base64.b64encode(image_bytes).decode("utf-8")

    return create_image_block(
        base64=data,
        mime_type=mime_type,
    )


def _normalize_image(
    image_bytes: bytes, filename: Path, guessed_mime: str | None
) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` in a format and size the API accepts.

    The format comes from the bytes, not the file name, so a ``.png`` that
    is really a PPM is converted rather than passed through.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width * height > MAX_DECODE_PIXELS:
                raise ImageReadError(
                    f"image is {width}x{height} px, too large to decode; "
                    f"downscale it to at most {MAX_IMAGE_SIDE_PX} px per "
                    "side first"
                )
            image_format = image.format or ""
            if (
                image_format in PASS_THROUGH_FORMATS
                and not getattr(image, "is_animated", False)
                and max(width, height) <= MAX_IMAGE_SIDE_PX
            ):
                # Decode it once (at most 2048x2048), so a truncated file,
                # e.g. a screenshot read while still being written, is
                # refused here rather than by the API.
                image.load()
                return image_bytes, PASS_THROUGH_FORMATS[image_format]
            return _to_png(image), "image/png"
    except ImageReadError:
        raise
    except Image.DecompressionBombError:
        raise ImageReadError(
            "image dimensions too large to decode; downscale it to at most "
            f"{MAX_IMAGE_SIDE_PX} px per side first"
        ) from None
    except UnidentifiedImageError:
        kind = guessed_mime or f"'{filename.suffix or filename.name}'"
        raise ImageReadError(
            f"unsupported or unreadable image format {kind}; "
            "convert it to PNG first"
        ) from None
    except (OSError, SyntaxError, ValueError):
        # Pillow raises these for a truncated or corrupt file of a format it
        # knows.
        raise ImageReadError(
            "the image data is truncated or corrupt; if the file is still "
            "being written, read it again once it is complete"
        ) from None


def _to_png(image: Image.Image) -> bytes:
    """Encode the first frame of ``image`` as PNG within the side limit."""
    image.seek(0)
    if image.mode in _HIGH_DEPTH_GRAY_MODES:
        # Downscale in float first, so the stretch below works on at most
        # 2048x2048 px rather than on another full-size float copy.
        image = image.convert("F")
        image.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))
        # Stretch to 8 bits; a plain convert("L") clips at 255, which turns
        # most 16-bit and float images white. A flat image has no range to
        # stretch: it is black if it is all zero (or below), else white.
        low, high = image.getextrema()
        if high > low:
            scale = 255.0 / (high - low)
            image = image.point(lambda v: (v - low) * scale)
        else:
            image = Image.new("F", image.size, 255.0 if low > 0 else 0.0)
        image = image.convert("L")
    else:
        if image.mode not in _PNG_MODES:
            # has_transparency_data, not an "A" band: LAB's A is a colour
            # channel, not alpha.
            image = image.convert(
                "RGBA" if image.has_transparency_data else "RGB"
            )
        # In place: the file's own image is not used again, and a copy of a
        # 100 Mpx canvas would cost another 400 MB.
        image.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
