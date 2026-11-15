"""Inbound webhook authentication."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status
from twilio.request_validator import RequestValidator

from app.core.dependencies import SettingsDep

logger = logging.getLogger(__name__)

TWILIO_SIGNATURE_HEADER = "X-Twilio-Signature"


def _public_request_url(request: Request, public_base_url: str | None) -> str:
    """Reconstruct the URL Twilio signed.

    Behind a TLS-terminating proxy or tunnel, ``request.url`` reflects the internal
    address, which would never match Twilio's signature.
    """
    if public_base_url is None:
        return str(request.url)
    url = public_base_url.rstrip("/") + request.url.path
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


async def verify_twilio_signature(request: Request, settings: SettingsDep) -> None:
    """Reject requests not signed by Twilio with this account's auth token."""
    if not settings.twilio_validate_signature:
        return

    signature = request.headers.get(TWILIO_SIGNATURE_HEADER)
    if not signature or settings.twilio_auth_token is None:
        logger.warning("Rejected Twilio webhook: missing signature")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature")

    form = await request.form()  # Starlette caches parsed form on the request.
    params = {key: value for key, value in form.multi_items() if isinstance(value, str)}
    base = str(settings.public_base_url) if settings.public_base_url else None
    url = _public_request_url(request, base)

    validator = RequestValidator(settings.twilio_auth_token.get_secret_value())
    if not validator.validate(url, params, signature):
        logger.warning("Rejected Twilio webhook: signature mismatch")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature")
