from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError
from synthetic import T0

from driftwatch.alerting import AlertManager, AlertRule, default_rules, load_rules
from driftwatch.types import MetricResult


class Recorder:
    def __init__(self, name="rec", fail=False):
        self.name = name
        self.fail = fail
        self.sent = []

    def send(self, alert):
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append(alert)


class Clock:
    def __init__(self):
        self.now = T0 + timedelta(hours=1)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


def result(minutes=60, feature="age", metric="psi", kind="drift", level="high", flagged=True, value=0.5, severity=None):
    ranks = {"none": 0.0, "low": 0.3, "medium": 0.55, "high": 0.8, "critical": 0.95}
    return MetricResult(
        model="churn",
        window_start=T0,
        window_end=T0 + timedelta(minutes=minutes),
        kind=kind,
        feature=feature,
        metric=metric,
        value=value,
        severity=ranks[level] if severity is None else severity,
        level=level,
        flagged=flagged,
    )


def make_manager(store, rules, notifiers=None, clock=None):
    notifiers = notifiers if notifiers is not None else {"rec": Recorder()}
    return AlertManager(store, rules, notifiers, clock or Clock())


def test_rule_selects_by_level_kind_and_filters():
    rule = AlertRule(name="r", kinds=["drift"], min_level="medium", metrics=["psi"], features=["age"])
    results = [
        result(level="high"),
        result(level="low"),
        result(level="high", metric="ks"),
        result(level="high", feature="income"),
        result(level="critical", kind="quality"),
        result(level="high", flagged=False),
    ]
    assert rule.select(results) == [results[0]]


def test_rule_rejects_unknown_level():
    with pytest.raises(ValidationError):
        AlertRule(name="r", min_level="severe")


def test_rule_model_matching():
    assert AlertRule(name="r").applies_to("anything")
    assert AlertRule(name="r", model="churn").applies_to("churn")
    assert not AlertRule(name="r", model="churn").applies_to("fraud")


def test_load_rules_from_yaml(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - name: pager\n    min_level: critical\n    notifiers: [slack]\n    cooldown_seconds: 60\n"
        "  - name: digest\n    kinds: [quality]\n    consecutive: 3\n"
    )
    rules = load_rules(path)
    assert [r.name for r in rules] == ["pager", "digest"]
    assert rules[0].notifiers == ["slack"]
    assert rules[1].consecutive == 3


def test_default_rules_cover_each_kind():
    kinds = {k for rule in default_rules() for k in rule.kinds}
    assert kinds == {"drift", "quality", "performance"}


def test_alert_is_raised_stored_and_delivered(store):
    recorder = Recorder()
    manager = make_manager(store, [AlertRule(name="drift-high", kinds=["drift"])], {"rec": recorder})
    raised = manager.process("churn", [result(), result(feature="income", level="critical", severity=0.95)])
    assert len(raised) == 1
    assert raised[0].level == "critical"
    assert "psi on income" in raised[0].message
    assert recorder.sent[0].rule == "drift-high"
    stored = store.list_alerts("churn")
    assert len(stored) == 1
    assert stored[0].delivered is True


def test_nothing_flagged_raises_nothing(store):
    manager = make_manager(store, default_rules())
    assert manager.process("churn", [result(flagged=False, level="none")]) == []
    assert manager.process("churn", []) == []
    assert store.list_alerts() == []


def test_cooldown_suppresses_repeats_until_it_expires(store):
    clock = Clock()
    manager = make_manager(store, [AlertRule(name="r", cooldown_seconds=600)], clock=clock)
    assert len(manager.process("churn", [result(60)])) == 1
    clock.advance(minutes=5)
    assert manager.process("churn", [result(65)]) == []
    clock.advance(minutes=10)
    assert len(manager.process("churn", [result(80)])) == 1
    assert len(store.list_alerts("churn")) == 2


def test_consecutive_windows_requirement(store):
    clock = Clock()
    rule = AlertRule(name="sustained", consecutive=3, cooldown_seconds=0)
    manager = make_manager(store, [rule], clock=clock)
    first = [result(60)]
    store.add_metrics(first)
    assert manager.process("churn", first) == []
    second = [result(120)]
    store.add_metrics(second)
    assert manager.process("churn", second) == []
    third = [result(180)]
    store.add_metrics(third)
    assert len(manager.process("churn", third)) == 1


def test_consecutive_resets_when_a_window_is_clean(store):
    rule = AlertRule(name="sustained", consecutive=2, cooldown_seconds=0)
    manager = make_manager(store, [rule])
    store.add_metrics([result(60, flagged=False, level="none")])
    current = [result(120)]
    store.add_metrics(current)
    assert manager.process("churn", current) == []


def test_failed_notifier_leaves_alert_undelivered_but_stored(store):
    good, bad = Recorder("good"), Recorder("bad", fail=True)
    manager = make_manager(store, [AlertRule(name="r")], {"good": good, "bad": bad})
    manager.process("churn", [result()])
    assert len(good.sent) == 1
    assert store.list_alerts("churn")[0].delivered is False


def test_rule_can_target_specific_notifiers(store):
    a, b = Recorder("a"), Recorder("b")
    manager = make_manager(store, [AlertRule(name="r", notifiers=["b"])], {"a": a, "b": b})
    manager.process("churn", [result()])
    assert a.sent == []
    assert len(b.sent) == 1


def test_message_truncates_long_lists(store):
    manager = make_manager(store, [AlertRule(name="r")])
    hits = [result(feature=f"f{i}", severity=0.8 - i * 0.01) for i in range(8)]
    alert = manager.process("churn", hits)[0]
    assert alert.message.endswith("and 3 more")
    assert alert.message.startswith("psi on f0")
