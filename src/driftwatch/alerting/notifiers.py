from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Callable, Protocol

import httpx

from driftwatch.config import Settings
from driftwatch.types import Alert

logger = logging.getLogger("driftwatch.alerts")

LEVEL_COLORS = {"low": "#c9a227", "medium": "#e08a1e", "high": "#d4472b", "critical": "#8b1a1a"}


class Notifier(Protocol):
    name: str

    def send(self, alert: Alert) -> None: ...


class LogNotifier:
    name = "log"

    def send(self, alert: Alert) -> None:
        logger.warning("[%s] %s %s: %s", alert.level, alert.model, alert.rule, alert.message)


class SlackNotifier:
    name = "slack"

    def __init__(self, webhook_url: str, client: httpx.Client | None = None, timeout: float = 5.0) -> None:
        self.webhook_url = webhook_url
        self._client = client or httpx.Client(timeout=timeout)

    def send(self, alert: Alert) -> None:
        payload = {
            "text": f"[{alert.level.upper()}] {alert.model}: {alert.rule}",
            "attachments": [
                {
                    "color": LEVEL_COLORS.get(alert.level, "#888888"),
                    "text": alert.message,
                    "footer": f"window ending {alert.window_end.isoformat()}",
                }
            ],
        }
        response = self._client.post(self.webhook_url, json=payload)
        response.raise_for_status()


class EmailNotifier:
    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        sender: str,
        recipients: list[str],
        username: str | None = None,
        password: str | None = None,
        starttls: bool = True,
        smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP,
    ) -> None:
        self.host = host
        self.port = port
        self.sender = sender
        self.recipients = recipients
        self.username = username
        self.password = password
        self.starttls = starttls
        self._factory = smtp_factory

    def send(self, alert: Alert) -> None:
        message = EmailMessage()
        message["Subject"] = f"[driftwatch] {alert.level} alert for {alert.model}: {alert.rule}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(f"{alert.message}\n\nwindow end: {alert.window_end.isoformat()}\n")
        with self._factory(self.host, self.port, timeout=10) as smtp:
            if self.starttls:
                smtp.starttls()
            if self.username and self.password:
                smtp.login(self.username, self.password)
            smtp.send_message(message)


def build_notifiers(settings: Settings) -> dict[str, Notifier]:
    notifiers: dict[str, Notifier] = {"log": LogNotifier()}
    if settings.slack_webhook_url:
        notifiers["slack"] = SlackNotifier(settings.slack_webhook_url)
    if settings.smtp_host and settings.smtp_to:
        notifiers["email"] = EmailNotifier(
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_from,
            [a.strip() for a in settings.smtp_to.split(",") if a.strip()],
            settings.smtp_username,
            settings.smtp_password,
        )
    return notifiers
