"""tasks.rules_version: the version of the control rules a task runs under

NULL, for every task created before this column existed, means version 1
(orchestrator/rules.py).

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-16

"""
from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("rules_version", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "rules_version")
