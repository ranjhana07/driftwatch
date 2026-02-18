from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("task_type", sa.String(16), nullable=False),
        sa.Column("positive_label", JSON_TYPE, nullable=False),
        sa.Column("config", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "prediction_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("features", JSON_TYPE, nullable=False),
        sa.Column("prediction", JSON_TYPE, nullable=False),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("embedding", JSON_TYPE, nullable=True),
        sa.Column("latency_ms", sa.Float, nullable=True),
    )
    op.create_index("ix_prediction_events_model_ts", "prediction_events", ["model", "ts"])
    op.create_table(
        "labels",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("label", JSON_TYPE, nullable=False),
        sa.Column("labeled_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_labels_model_labeled_at", "labels", ["model", "labeled_at"])
    op.create_table(
        "reference_sets",
        sa.Column("model", sa.String(64), primary_key=True),
        sa.Column("payload", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "metric_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("window_start", sa.DateTime, nullable=False),
        sa.Column("window_end", sa.DateTime, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("feature", sa.String(128), nullable=False),
        sa.Column("metric", sa.String(48), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("p_adjusted", sa.Float, nullable=True),
        sa.Column("severity", sa.Float, nullable=False),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("flagged", sa.Boolean, nullable=False),
        sa.Column("n_ref", sa.Integer, nullable=False),
        sa.Column("n_cur", sa.Integer, nullable=False),
    )
    op.create_index("ix_metric_results_model_window", "metric_results", ["model", "window_end"])
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("rule", sa.String(64), nullable=False),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("window_end", sa.DateTime, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("delivered", sa.Boolean, nullable=False),
    )
    op.create_index("ix_alerts_model_created", "alerts", ["model", "created_at"])


def downgrade() -> None:
    for table in ("alerts", "metric_results", "reference_sets", "labels", "prediction_events", "models"):
        op.drop_table(table)
