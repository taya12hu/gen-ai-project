"""
Authentication - transactional email delivery.

Sends transactional email via Resend's REST API (https://resend.com).
Only used for password reset links today - kept as its own module so the
reset flow doesn't need to know which provider is behind it. Resend's
sandbox sender (the default RESEND_FROM_EMAIL below) only delivers to the
email address the Resend account itself was signed up with, until a custom
domain is verified - fine for development/demo use, not for real end users
until a domain is verified in the Resend dashboard.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


class EmailSendError(Exception):
    pass


def send_password_reset_email(to_email: str, reset_url: str) -> None:
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        raise EmailSendError("RESEND_API_KEY is not configured")
    from_email = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")

    logger.info("Sending password reset email to %s", to_email)
    response = httpx.post(
        RESEND_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "from": from_email,
            "to": [to_email],
            "subject": "Reset your password",
            "html": (
                "<p>Someone requested a password reset for this account.</p>"
                f'<p><a href="{reset_url}">Click here to reset your password</a>. '
                "This link expires in 30 minutes.</p>"
                "<p>If you didn't request this, you can safely ignore this email.</p>"
            ),
        },
        timeout=10.0,
    )
    if response.status_code >= 400:
        logger.warning("Resend API error %d sending to %s: %s", response.status_code, to_email, response.text)
        raise EmailSendError(f"Resend API error {response.status_code}: {response.text}")
