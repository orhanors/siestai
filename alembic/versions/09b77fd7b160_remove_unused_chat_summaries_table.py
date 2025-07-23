"""Remove unused chat_summaries table

Revision ID: 09b77fd7b160
Revises: 571e91790422
Create Date: 2025-07-23 19:37:36.165753

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '09b77fd7b160'
down_revision: Union[str, Sequence[str], None] = '571e91790422'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop chat_summaries table and its indexes only
    op.drop_index('idx_chat_summaries_session', table_name='chat_summaries')
    op.drop_index('idx_chat_summaries_type', table_name='chat_summaries')
    op.drop_index('idx_chat_summaries_user_profile', table_name='chat_summaries')
    op.drop_table('chat_summaries')


def downgrade() -> None:
    """Downgrade schema."""
    # Recreate chat_summaries table if needed for rollback
    op.create_table('chat_summaries',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.TEXT(), nullable=False),
    sa.Column('profile_id', sa.TEXT(), nullable=False),
    sa.Column('summary_type', sa.TEXT(), nullable=False),
    sa.Column('title', sa.TEXT(), nullable=True),
    sa.Column('summary_content', sa.TEXT(), nullable=False),
    sa.Column('key_topics', postgresql.ARRAY(sa.TEXT()), nullable=True, comment='Array of key topics discussed'),
    sa.Column('message_range_start', sa.INTEGER(), nullable=True, comment='First message order in summary'),
    sa.Column('message_range_end', sa.INTEGER(), nullable=True, comment='Last message order in summary'),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True, comment='Summary embedding'),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.CheckConstraint("summary_type IN ('session', 'topic', 'periodic')", name='check_summary_type'),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Recreate indexes
    op.create_index('idx_chat_summaries_session', 'chat_summaries', ['session_id'])
    op.create_index('idx_chat_summaries_type', 'chat_summaries', ['summary_type'])
    op.create_index('idx_chat_summaries_user_profile', 'chat_summaries', ['user_id', 'profile_id'])
