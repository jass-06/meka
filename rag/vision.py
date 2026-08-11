"""
rag/vision.py

Turns an image into a text caption via the HuggingFace Inference API so it
can be folded into the same text-embedding / retrieval pipeline as
documents. This keeps the system multimodal without needing a separate
image vector space.
"""
import base64
from typing import Optional

import requests

from rag.config import Config


class VisionError(RuntimeError):
    pass


class ImageCaptioner:
    def __init__(self):
        self.url = Config.HF_VISION_URL
        self.token = Config.HF_API_TOKEN
        self.timeout = Config.REQUEST_TIMEOUT

    def is_available(self) -> bool:
        return bool(self.token)

    def caption(self, image_base64: str) -> Optional[str]:
        """Return a short text caption for a base64-encoded image, or None on failure."""
        if not self.is_available():
            return None
        try:
            image_bytes = base64.b64decode(image_base64)
            resp = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                data=image_bytes,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return None
        except Exception:
            # Vision is best-effort; never let a captioning failure break chat.
            return None
