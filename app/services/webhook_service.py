"""
Greenpack Pro — Webhook Service
Posts inspection results to external ERP/QMS systems via HMAC-signed HTTP
"""
import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime

import httpx

log = logging.getLogger(__name__)


async def post_webhook(
    result: dict,
    webhook_url: str,
    secret: str,
    max_retries: int = 3,
) -> bool:
    """
    POST inspection result to configured webhook URL.
    Signs request with HMAC-SHA256 for security.
    Retries 3 times with exponential backoff.
    """
    payload = {
        "event": "inspection.completed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "job_id": result.get("job_id"),
        "job_ref": result.get("job_ref"),
        "client": result.get("client_name"),
        "product": result.get("product_name"),
        "pass_fail": result.get("pass_fail"),
        "overall_score": result.get("overall_score"),
        "ocr_errors_count": len(result.get("text_errors", [])),
        "color_pass": all(z.get("pass", True) for z in result.get("color_results", [])),
        "barcode_pass": all(b.get("pass", True) for b in result.get("barcode_results", [])),
        "defect_count": len(result.get("defects", [])),
        "processing_time_ms": result.get("processing_time_ms"),
    }

    body = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        (secret or "no-secret").encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Greenpack-Signature": f"sha256={signature}",
        "X-Greenpack-Version": "1.0",
        "User-Agent": "GreenpackPro/1.0",
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(webhook_url, content=body, headers=headers)
                if response.status_code < 300:
                    log.info(f"Webhook delivered: {response.status_code}")
                    return True
                log.warning(
                    f"Webhook attempt {attempt+1}/{max_retries} failed: "
                    f"HTTP {response.status_code}"
                )
        except Exception as e:
            log.warning(f"Webhook attempt {attempt+1}/{max_retries} error: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

    log.error(f"Webhook delivery failed after {max_retries} attempts")
    return False
