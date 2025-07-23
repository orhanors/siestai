# Chat History Database Schema

This document outlines the database schema for chat history management with support for users and profiles.

## Tables

### 1. chat_sessions

Stores chat session information.

```sql
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,                  -- External user identifier
    profile_id TEXT NOT NULL,               -- Profile identifier within user
    session_name TEXT,                      -- Optional session name
    session_metadata JSONB DEFAULT '{}',   -- Additional session metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Indexes for efficient queries
CREATE INDEX idx_chat_sessions_user_profile ON chat_sessions(user_id, profile_id);
CREATE INDEX idx_chat_sessions_created_at ON chat_sessions(created_at DESC);
CREATE INDEX idx_chat_sessions_active ON chat_sessions(is_active) WHERE is_active = true;
```

### 2. chat_messages

Stores individual chat messages with embeddings for semantic search.

```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,                  -- Denormalized for faster queries
    profile_id TEXT NOT NULL,               -- Denormalized for faster queries
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    message_metadata JSONB DEFAULT '{}',   -- Sources, references, etc.
    embedding VECTOR(1536),                -- Message embedding for semantic search
    token_count INTEGER,                   -- Token count for the message
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    message_order INTEGER NOT NULL        -- Order within session
);

-- Indexes for efficient queries
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id, message_order);
CREATE INDEX idx_chat_messages_user_profile ON chat_messages(user_id, profile_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX idx_chat_messages_role ON chat_messages(role);

-- Vector similarity search index
CREATE INDEX idx_chat_messages_embedding ON chat_messages 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 3. chat_summaries (REMOVED)

*Note: The chat_summaries table has been removed as it was not being used in the current implementation. The system now relies on direct message history and semantic search for memory retrieval.*

## Memory Types

### Short-term Memory
- **Recent Messages**: Last N messages in the current session
- **Session Context**: Current session's conversation flow
- **Immediate References**: Recently mentioned documents/topics

### Long-term Memory
- **Historical Context**: Relevant past conversations via semantic search
- **User Preferences**: Learned preferences and patterns from message history
- **Cross-session Memory**: Access to messages from previous sessions

## Usage Examples

### Create a new chat session
```sql
INSERT INTO chat_sessions (user_id, profile_id, session_name)
VALUES ('user_123', 'work_profile', 'AI Research Discussion');
```

### Add a message to session
```sql
INSERT INTO chat_messages (
    session_id, user_id, profile_id, role, content, 
    embedding, message_order
)
VALUES (
    'session_uuid', 'user_123', 'work_profile', 'user', 
    'What is LangGraph?', '[0.1, 0.2, ...]'::vector, 1
);
```

### Search similar messages across user's history
```sql
SELECT content, session_id, created_at,
       embedding <=> $1::vector AS similarity
FROM chat_messages 
WHERE user_id = $2 AND profile_id = $3 
  AND embedding IS NOT NULL
  AND (embedding <=> $1::vector) < 0.8  -- similarity threshold
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

### Get recent session context
```sql
SELECT role, content, message_metadata, created_at
FROM chat_messages 
WHERE session_id = $1 
ORDER BY message_order DESC 
LIMIT 20;
```

## Memory Management Strategy

1. **Short-term (Current Session)**
   - Keep last 10-20 messages in active memory
   - Include current session context in LangGraph state

2. **Medium-term (Recent Sessions)**
   - Search recent sessions for relevant context
   - Use semantic search on message embeddings
   - Include recent messages from current session for continuity

3. **Long-term (Historical)**
   - Semantic search across all message history
   - Topic-based retrieval using message embeddings
   - Cross-session memory access for user preferences

4. **Cleanup Strategy**
   - Archive old sessions after N days
   - Mark inactive sessions as `is_active = false`
   - Maintain message history for semantic search