"""memories table with pgvector embeddings

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-01

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memories",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("task_id", sa.String(32), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        # No dimension: there is no ANN index to size, and memories are ranked by
        # exact distance after filtering one user's rows.
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_memories_user_kind", "memories", ["user_id", "kind"])


def downgrade() -> None:
    op.drop_index("ix_memories_user_kind", table_name="memories")
    op.drop_table("memories")
    # The vector extension stays: dropping it would fail if anything else used it.
