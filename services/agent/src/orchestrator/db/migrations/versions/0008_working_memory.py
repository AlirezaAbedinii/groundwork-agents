"""working_memory table for the Postgres working-memory backend

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-16

"""
from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "working_memory",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("task_id", sa.String(32), nullable=False),
        # NULL marks an error row; the unique constraint treats NULLs as distinct.
        sa.Column("field", sa.String(255), nullable=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", "field", name="uq_working_memory_task_field"),
    )
    op.create_index("ix_working_memory_expires_at", "working_memory", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_working_memory_expires_at", table_name="working_memory")
    op.drop_table("working_memory")
