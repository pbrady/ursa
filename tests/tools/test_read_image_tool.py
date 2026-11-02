import base64
import io
import os
import random
import subprocess
from pathlib import Path

import pytest
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image, ImageDraw
from pydantic import BaseModel

from tests.tools.utils import make_runtime
from ursa.cli.config import ChatModelConfig
from ursa.tools.read_image_tool import (
    MAX_IMAGE_SIDE_PX,
    ImageReadError,
    image_block_from_file,
    read_image_tool,
)


class DigitAnswer(BaseModel):
    number: int


def _write_digit_image(path: Path, number: int) -> None:
    size = 128
    target_extent = int(size * 0.9)
    text = str(number)
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)

    font_size = 1
    for candidate in range(1, size * 2):
        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font_size=candidate
        )
        if (right - left) > target_extent or (bottom - top) > target_extent:
            break
        font_size = candidate

    left, top, right, bottom = draw.textbbox((0, 0), text, font_size=font_size)
    x = (size - (right - left)) // 2 - left
    y = (size - (bottom - top)) // 2 - top
    draw.text((x, y), text, fill="black", font_size=font_size)
    image.save(path)


def _require_openai_model(model: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip(f"Requires OPENAI_API_KEY for {model}")


def _require_ollama_model(model: str) -> None:
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Ollama is not available for {model}: {exc}")

    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        pytest.skip(f"Ollama model {model!r} is not available: {message}")


@pytest.fixture(
    params=[
        pytest.param(
            {
                "model": "openai:gpt-5.2",
                "max_completion_tokens": 128,
                "use_responses_api": True,
            },
            id="openai-gpt-5.2",
        ),
        # Add other ChatModelConfig values for local testing here
    ]
)
def vision_chat_model(request, monkeypatch) -> BaseChatModel:
    kwargs = dict(request.param)
    model = kwargs["model"]
    if model.startswith("openai:"):
        _require_openai_model(model.split(":", 1)[1])
    elif model.startswith("ollama:"):
        _require_ollama_model(model.split(":", 1)[1])

    return ChatModelConfig(**kwargs).init_chat_model()


def test_image_block_from_file_returns_base64_image_block(tmp_path: Path):
    target = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(target)

    result = image_block_from_file(target)

    assert result["type"] == "image"
    assert result["mime_type"] == "image/png"
    # file_id is reserved for provider-side file-store references and must
    # never carry a local path (providers may prioritize it over base64)
    assert "file_id" not in result
    assert base64.b64decode(result["base64"]).startswith(b"\x89PNG")


def test_image_block_from_file_rejects_large_images(tmp_path: Path):
    target = tmp_path / "large.png"
    max_image_mb = 2
    max_image_bytes = max_image_mb * 1024 * 1024
    with target.open("wb") as image_file:
        image_file.seek(max_image_bytes)
        image_file.write(b"\0")

    with pytest.raises(ValueError, match="File too large"):
        image_block_from_file(target, max_size_mb=max_image_mb)


def test_read_image_tool_reads_from_runtime_workspace(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "logo.png"
    Image.new("RGB", (2, 1), color=(0, 128, 255)).save(target)

    result = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-image-call",
        ),
    )

    assert len(result) == 2
    assert result[0] == {
        "type": "text",
        "text": f"Image file: {target.name}",
    }
    assert result[1]["type"] == "image"
    assert result[1]["mime_type"] == "image/png"
    assert "file_id" not in result[1]
    assert base64.b64decode(result[1]["base64"]).startswith(b"\x89PNG")


@pytest.mark.real_llm
def test_read_image_tool_output_can_be_read_by_attached_llm(
    tmp_path: Path, vision_chat_model: BaseChatModel
):
    number = random.SystemRandom().randint(10, 99)
    target = tmp_path / "number.png"
    _write_digit_image(target, number)

    tool_llm = vision_chat_model.bind_tools([read_image_tool])
    structured_llm = tool_llm.with_structured_output(
        DigitAnswer,
        method="function_calling",
    )
    tool_call_id = "call_read_number_image"
    image_blocks = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=tool_llm,
            tool_call_id=tool_call_id,
        ),
    )

    response = structured_llm.invoke([
        HumanMessage(
            content=(
                "Use the attached read_image_tool result to identify the "
                "two-digit number shown in number.png."
            )
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_image_tool",
                    "args": {"image_path": target.name},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=image_blocks,
            name="read_image_tool",
            tool_call_id=tool_call_id,
        ),
    ])

    assert response.number == number


def test_image_block_from_file_converts_svg_to_png(tmp_path: Path):
    target = tmp_path / "circle.svg"

    target.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="100"
             height="100"
             viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="red"/>
        </svg>
        """,
        encoding="utf-8",
    )

    result = image_block_from_file(target)

    assert result["type"] == "image"
    assert result["mime_type"] == "image/png"

    decoded = base64.b64decode(result["base64"])
    assert decoded.startswith(b"\x89PNG")


def test_svg_conversion_labels_original_filename(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "plot.svg"

    target.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg"
             width="20"
             height="20">
          <rect width="20" height="20" fill="blue"/>
        </svg>
        """,
        encoding="utf-8",
    )

    result = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-svg-call",
        ),
    )

    # The label keeps the original .svg name even though the payload is png
    assert result[0] == {"type": "text", "text": "Image file: plot.svg"}
    assert result[1]["mime_type"] == "image/png"
    assert "file_id" not in result[1]
    assert base64.b64decode(result[1]["base64"]).startswith(b"\x89PNG")


def test_image_block_from_file_passes_png_through_unchanged(tmp_path: Path):
    target = tmp_path / "pixel.png"
    Image.new("RGB", (3, 2), color=(0, 255, 0)).save(target)

    result = image_block_from_file(target)

    assert result["mime_type"] == "image/png"
    assert base64.b64decode(result["base64"]) == target.read_bytes()


@pytest.mark.parametrize(
    ("suffix", "mime_type"),
    [(".jpg", "image/jpeg"), (".gif", "image/gif"), (".webp", "image/webp")],
)
def test_image_block_from_file_passes_accepted_formats_through(
    tmp_path: Path, suffix: str, mime_type: str
):
    target = tmp_path / f"pixel{suffix}"
    Image.new("RGB", (3, 2), color=(0, 0, 255)).save(target)

    result = image_block_from_file(target)

    assert result["mime_type"] == mime_type
    assert base64.b64decode(result["base64"]) == target.read_bytes()


@pytest.mark.parametrize("suffix", [".ppm", ".pgm", ".pbm", ".bmp", ".tiff"])
def test_image_block_from_file_converts_other_rasters_to_png(
    tmp_path: Path, suffix: str
):
    # e.g. the .ppm screenshots QEMU's VNC writes, which OpenAI answers with
    # a 400 "does not represent a valid image" when sent as image/x-portable-*
    target = tmp_path / f"screen{suffix}"
    mode = {".pgm": "L", ".pbm": "1"}.get(suffix, "RGB")
    Image.new(mode, (5, 4), color=1).save(target)

    result = image_block_from_file(target)

    assert result["mime_type"] == "image/png"
    decoded = Image.open(io.BytesIO(base64.b64decode(result["base64"])))
    assert decoded.format == "PNG"
    assert decoded.size == (5, 4)


def test_image_block_from_file_converts_ppm_named_png(tmp_path: Path):
    # The format comes from the bytes, not the suffix
    target = tmp_path / "screen.png"
    Image.new("RGB", (5, 4), color=(9, 9, 9)).save(target, format="PPM")

    result = image_block_from_file(target)

    assert result["mime_type"] == "image/png"
    assert base64.b64decode(result["base64"]).startswith(b"\x89PNG")


def test_image_block_from_file_downscales_large_images(tmp_path: Path):
    target = tmp_path / "wide.png"
    Image.new("RGB", (MAX_IMAGE_SIDE_PX * 2, 100)).save(target)

    result = image_block_from_file(target)

    decoded = Image.open(io.BytesIO(base64.b64decode(result["base64"])))
    assert result["mime_type"] == "image/png"
    assert decoded.size == (MAX_IMAGE_SIDE_PX, 50)


def test_image_block_from_file_converts_animated_gif(tmp_path: Path):
    target = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (4, 4), color=c) for c in ("red", "blue")]
    frames[0].save(target, save_all=True, append_images=frames[1:])

    result = image_block_from_file(target)

    assert result["mime_type"] == "image/png"


def test_image_block_from_file_refuses_unknown_format(tmp_path: Path):
    target = tmp_path / "data.xyz"
    target.write_bytes(b"not an image at all")

    with pytest.raises(ImageReadError, match="convert it to PNG first"):
        image_block_from_file(target)


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("data.xyz", b"not an image", "unsupported or unreadable image format"),
        ("broken.png", b"\x89PNG garbage", "image/png; convert it to PNG"),
        ("missing.png", None, "file not found"),
    ],
)
def test_read_image_tool_returns_text_error_for_unreadable_files(
    tmp_path: Path,
    chat_model: BaseChatModel,
    name: str,
    content: bytes | None,
    expected: str,
):
    if content is not None:
        (tmp_path / name).write_bytes(content)

    result = read_image_tool.func(
        image_path=name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-bad-image-call",
        ),
    )

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"].startswith(f"Could not read image file {name}: ")
    assert expected in result[0]["text"]


def test_read_image_tool_sends_ppm_as_png(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "vnc.ppm"
    Image.new("RGB", (6, 3), color=(1, 2, 3)).save(target)

    result = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-ppm-call",
        ),
    )

    assert result[0] == {"type": "text", "text": "Image file: vnc.ppm"}
    assert result[1]["mime_type"] == "image/png"
    assert base64.b64decode(result[1]["base64"]).startswith(b"\x89PNG")


def _decoded(block) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(block["base64"])))
    image.load()
    return image


@pytest.mark.parametrize(
    ("samples", "extrema"),
    [
        # 16-bit PGM using only 0..1000 of 0..65535: stretched, not near-black
        ([0, 250, 500, 1000], (0, 255)),
        # a flat white 16-bit frame stays white
        ([65535] * 4, (255, 255)),
        ([0] * 4, (0, 0)),
    ],
)
def test_image_block_from_file_stretches_16_bit_grayscale(
    tmp_path: Path, samples: list[int], extrema: tuple[int, int]
):
    target = tmp_path / "depth.pgm"
    target.write_bytes(
        b"P5\n2 2\n65535\n" + b"".join(v.to_bytes(2, "big") for v in samples)
    )

    image = _decoded(image_block_from_file(target))

    assert (image.format, image.mode) == ("PNG", "L")
    assert image.getextrema() == extrema


def test_image_block_from_file_does_not_take_lab_a_band_for_alpha(
    tmp_path: Path,
):
    target = tmp_path / "lab.tiff"
    Image.new("LAB", (3, 2), color=(50, 0, 0)).save(target)

    assert _decoded(image_block_from_file(target)).mode == "RGB"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_image_block_from_file_refuses_truncated_pass_through_image(
    tmp_path: Path, fmt: str
):
    # e.g. a screenshot read while it is still being written: the header
    # parses, so it must be decoded to be caught before the API sees it
    target = tmp_path / f"screen.{fmt.lower()}"
    Image.effect_noise((256, 256), 50).convert("RGB").save(target, format=fmt)
    target.write_bytes(target.read_bytes()[:3000])

    with pytest.raises(ImageReadError, match="truncated or corrupt"):
        image_block_from_file(target)


def test_read_image_tool_returns_text_error_for_a_directory(
    tmp_path: Path, chat_model: BaseChatModel
):
    (tmp_path / "shots.png").mkdir()

    result = read_image_tool.func(
        image_path="shots.png",
        runtime=make_runtime(
            tmp_path, llm=chat_model, tool_call_id="read-dir-call"
        ),
    )

    assert result == [
        {
            "type": "text",
            "text": "Could not read image file shots.png: cannot read file: "
            "Is a directory",
        }
    ]
