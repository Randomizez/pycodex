"""Image preparation helpers for the Python Codex prototype.

Original Codex mapping:
- Corresponds to `codex-rs/utils/image/src/lib.rs`.

Expected behavior:
- Resize images down to `MAX_DIMENSION` on the longest side before they are
  attached to a model request, matching upstream `PromptImageMode::ResizeToFit`.
- Keep the original bytes when the caller asks for `original` detail or the
  image already fits.
"""

import base64
import io
import mimetypes
from pathlib import Path

from PIL import Image

import typing

MAX_DIMENSION = 2048

_PRESERVABLE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp")


class ImageProcessingError(RuntimeError):
    pass


def load_image_data_url(
    path: 'Path',
    resize_to_fit: 'bool' = True,
) -> 'str':
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ImageProcessingError(
            "`{0}` does not look like an image file".format(path)
        )

    image_bytes = path.read_bytes()
    if resize_to_fit:
        mime_type, image_bytes = _resize_to_fit(mime_type, image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return "data:{0};base64,{1}".format(mime_type, encoded)


def _resize_to_fit(
    mime_type: 'str',
    image_bytes: 'bytes',
) -> 'typing.Tuple[str, bytes]':
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
            return mime_type, image_bytes

        scale = float(MAX_DIMENSION) / float(max(width, height))
        target_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        resized = image.resize(target_size, Image.BILINEAR)
        target_mime = (
            mime_type if mime_type in _PRESERVABLE_MIME_TYPES else "image/png"
        )
        if target_mime == "image/jpeg":
            resized = resized.convert("RGB")
            save_format, save_kwargs = "JPEG", {"quality": 85}
        elif target_mime == "image/webp":
            resized = resized.convert("RGBA")
            save_format, save_kwargs = "WEBP", {"lossless": True}
        else:
            resized = resized.convert("RGBA")
            save_format, save_kwargs = "PNG", {}

        buffer = io.BytesIO()
        resized.save(buffer, format=save_format, **save_kwargs)

    return target_mime, buffer.getvalue()
