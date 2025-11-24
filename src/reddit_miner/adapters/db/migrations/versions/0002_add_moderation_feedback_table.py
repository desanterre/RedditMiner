"""update comments table for AI moderation

Revision ID: a2b529603882
Revises: 0001
Create Date: 2025-11-24 00:50:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_ai_moderation'
down_revision = '0001'
branch_labels = None
depends_on = None

def upgrade():
    # Add columns for AI moderation
    op.add_column('comments', sa.Column('ai_label', sa.String(20), nullable=True))        # EXTREMIST / TOXIC / SAFE
    op.add_column('comments', sa.Column('ai_explanation', sa.Text, nullable=True))        # reasoning by AI
    op.add_column('comments', sa.Column('ai_model', sa.String(100), nullable=True))       # model name/version
    op.add_column('comments', sa.Column('ai_moderated_at', sa.DateTime, server_default=sa.text('NOW()'), nullable=True))

def downgrade():
    op.drop_column('comments', 'ai_label')
    op.drop_column('comments', 'ai_explanation')
    op.drop_column('comments', 'ai_model')
    op.drop_column('comments', 'ai_moderated_at')
