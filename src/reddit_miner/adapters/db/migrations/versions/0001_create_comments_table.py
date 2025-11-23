"""create comments table"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "comments",
        sa.Column("comment_id", sa.Text(), primary_key=True),
        sa.Column("author", sa.Text()),
        sa.Column("subreddit", sa.Text()),
        sa.Column("post_id", sa.Text()),
        sa.Column("post_title", sa.Text()),
        sa.Column("score", sa.Integer()),
        sa.Column("num_comments", sa.Integer()),
        sa.Column("date_utc", sa.DateTime()),
        sa.Column("permalink", sa.Text()),
        sa.Column("body", sa.Text()),
    )

def downgrade():
    op.drop_table("comments")
