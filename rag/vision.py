"""
Image captioning via the HuggingFace Inference API. Used both at ingest
time (to turn an uploaded image into searchable text) and at query time
(to let a user attach an image to a question).

Fails soft: if the HF token is missing or the API call errors out, we
return a placeholder caption instead of crashing the pipeline.
"""
import logging

import requests

from rag import config

logger = logging.getLogger(__name__)

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{config.VISION_MODEL}"


def caption_image(image_bytes: bytes) -> str:
    if not config.HF_API_TOKEN:
        logger.warning("HF_API_TOKEN not set; skipping image captioning.")
        return "[Image uploaded — captioning unavailable, no HF_API_TOKEN configured]"

    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    try:
        response = requests.post(
            HF_INFERENCE_URL, headers=headers, data=image_bytes, timeout=30
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return "[Image uploaded — no caption returned by vision model]"
    except Exception as exc:  # noqa: BLE001 - we want this to degrade gracefully
        logger.error("Vision captioning failed: %s", exc)
        return "[Image uploaded — captioning failed]"
