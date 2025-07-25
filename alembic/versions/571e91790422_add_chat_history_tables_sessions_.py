"""Add chat history tables (sessions, messages, summaries)

Revision ID: 571e91790422
Revises: 9fe4a39052db
Create Date: 2025-07-23 19:22:31.395093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '571e91790422'
down_revision: Union[str, Sequence[str], None] = '9fe4a39052db'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('chat_sessions',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False, comment='External user identifier'),
    sa.Column('profile_id', sa.Text(), nullable=False, comment='Profile identifier within user'),
    sa.Column('session_name', sa.Text(), nullable=True, comment='Optional session name'),
    sa.Column('session_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'"), nullable=False, comment='Additional session metadata'),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chat_messages',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False, comment='Denormalized for faster queries'),
    sa.Column('profile_id', sa.Text(), nullable=False, comment='Denormalized for faster queries'),
    sa.Column('role', sa.Text(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('message_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'"), nullable=False, comment='Sources, references, etc.'),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True, comment='Message embedding for semantic search'),
    sa.Column('token_count', sa.Integer(), nullable=True, comment='Token count for the message'),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.Column('message_order', sa.Integer(), nullable=False, comment='Order within session'),
    sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name='check_message_role'),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chat_summaries',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('profile_id', sa.Text(), nullable=False),
    sa.Column('summary_type', sa.Text(), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('summary_content', sa.Text(), nullable=False),
    sa.Column('key_topics', postgresql.ARRAY(sa.Text()), nullable=True, comment='Array of key topics discussed'),
    sa.Column('message_range_start', sa.Integer(), nullable=True, comment='First message order in summary'),
    sa.Column('message_range_end', sa.Integer(), nullable=True, comment='Last message order in summary'),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True, comment='Summary embedding'),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    sa.CheckConstraint("summary_type IN ('session', 'topic', 'periodic')", name='check_summary_type'),
    sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better performance
    op.create_index('idx_chat_sessions_user_profile', 'chat_sessions', ['user_id', 'profile_id'])
    op.create_index('idx_chat_sessions_created_at', 'chat_sessions', ['created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_chat_sessions_active', 'chat_sessions', ['is_active'], postgresql_where=sa.text('is_active = true'))
    
    op.create_index('idx_chat_messages_session', 'chat_messages', ['session_id', 'message_order'])
    op.create_index('idx_chat_messages_user_profile', 'chat_messages', ['user_id', 'profile_id'])
    op.create_index('idx_chat_messages_created_at', 'chat_messages', ['created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_chat_messages_role', 'chat_messages', ['role'])
    
    op.create_index('idx_chat_summaries_session', 'chat_summaries', ['session_id'])
    op.create_index('idx_chat_summaries_user_profile', 'chat_summaries', ['user_id', 'profile_id'])
    op.create_index('idx_chat_summaries_type', 'chat_summaries', ['summary_type'])
    
    # Create trigger function to auto-update session activity
    op.execute("""
        CREATE OR REPLACE FUNCTION update_session_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE chat_sessions 
            SET updated_at = NOW() 
            WHERE id = NEW.session_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger to auto-update session activity on new messages
    op.execute("""
        CREATE TRIGGER trigger_update_session_activity
            AFTER INSERT ON chat_messages
            FOR EACH ROW
            EXECUTE FUNCTION update_session_updated_at();
    """)
    
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_session_activity ON chat_messages")
    op.execute("DROP FUNCTION IF EXISTS update_session_updated_at() CASCADE")
    
    # Drop indexes
    op.drop_index('idx_chat_summaries_type', 'chat_summaries')
    op.drop_index('idx_chat_summaries_user_profile', 'chat_summaries')
    op.drop_index('idx_chat_summaries_session', 'chat_summaries')
    
    op.drop_index('idx_chat_messages_role', 'chat_messages')
    op.drop_index('idx_chat_messages_created_at', 'chat_messages')
    op.drop_index('idx_chat_messages_user_profile', 'chat_messages')
    op.drop_index('idx_chat_messages_session', 'chat_messages')
    
    op.drop_index('idx_chat_sessions_active', 'chat_sessions')
    op.drop_index('idx_chat_sessions_created_at', 'chat_sessions')
    op.drop_index('idx_chat_sessions_user_profile', 'chat_sessions')
    
    # Drop tables
    op.drop_table('chat_summaries')
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    # ### end Alembic commands ###
