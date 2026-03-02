from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from driftwatch.alerting.notifiers import EmailNotifier, LogNotifier, SlackNotifier, build_notifiers
from driftwatch.config import Settings
from driftwatch.types import Alert

ALERT = Alert(
    id=1,
    model="churn",
    rule="drift-high",
    level="high",
    message="psi on age = 0.43 (high)",
    window_end=datetime(2026, 3, 1, 11, tzinfo=timezone.utc),
    created_at=datetime(2026, 3, 1, 11, 1, tzinfo=timezone.utc),
)


def test_slack_posts_json_to_webhook():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="ok")

    notifier = SlackNotifier("https://hooks.example/abc", httpx.Client(transport=httpx.MockTransport(handler)))
    notifier.send(ALERT)
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://hooks.example/abc"
    assert "HIGH" in body["text"] and "churn" in body["text"]
    assert body["attachments"][0]["text"] == ALERT.message


def test_slack_raises_on_http_error():
    notifier = SlackNotifier("https://hooks.example/abc", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(httpx.HTTPStatusError):
        notifier.send(ALERT)


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.actions = []
        self.message = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.actions.append("starttls")

    def login(self, user, password):
        self.actions.append(("login", user, password))

    def send_message(self, message):
        self.message = message


def test_email_sends_message_with_tls_and_login():
    FakeSMTP.instances.clear()
    notifier = EmailNotifier(
        "smtp.example", 587, "alerts@example", ["a@example", "b@example"], "user", "pw", smtp_factory=FakeSMTP
    )
    notifier.send(ALERT)
    smtp = FakeSMTP.instances[0]
    assert smtp.actions == ["starttls", ("login", "user", "pw")]
    assert smtp.message["To"] == "a@example, b@example"
    assert "high alert for churn" in smtp.message["Subject"]
    assert "psi on age" in smtp.message.get_content()


def test_email_without_credentials_skips_login():
    FakeSMTP.instances.clear()
    EmailNotifier("smtp.example", 25, "x@y", ["z@y"], starttls=False, smtp_factory=FakeSMTP).send(ALERT)
    assert FakeSMTP.instances[0].actions == []


def test_log_notifier_writes_a_warning(caplog):
    with caplog.at_level("WARNING", logger="driftwatch.alerts"):
        LogNotifier().send(ALERT)
    assert "drift-high" in caplog.text


def test_build_notifiers_from_settings():
    bare = build_notifiers(Settings(database_url=None, slack_webhook_url=None, smtp_host=None))
    assert set(bare) == {"log"}
    full = build_notifiers(
        Settings(slack_webhook_url="https://hooks.example/x", smtp_host="smtp.example", smtp_to="a@x, b@x")
    )
    assert set(full) == {"log", "slack", "email"}
    assert full["email"].recipients == ["a@x", "b@x"]
