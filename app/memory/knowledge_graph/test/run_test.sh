#!/bin/bash

# Knowledge Graph Ingestion Test Runner
# This script runs the KG ingestion test with proper environment setup

echo "🚀 Starting Knowledge Graph Ingestion Test"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "test_ingest_kg.py" ]; then
    echo "❌ Error: Please run this script from the app/memory/knowledge_graph/test/ directory"
    exit 1
fi

# Check if Docker containers are running
echo "🔍 Checking if Neo4j is running..."
if ! docker ps | grep -q neo4j; then
    echo "⚠️  Neo4j container not found running. Starting with docker-compose..."
    echo "Please make sure you have docker-compose.yaml configured and run:"
    echo "   docker-compose up -d neo4j"
    echo ""
    echo "Continuing with test (assuming Neo4j is running locally)..."
fi

# Check if PostgreSQL is running (for completeness)
echo "🔍 Checking if PostgreSQL is running..."
if ! docker ps | grep -q postgres; then
    echo "⚠️  PostgreSQL container not found running. You may want to start it:"
    echo "   docker-compose up -d pgvector"
fi

# Check if .env file exists
if [ ! -f "../../../../.env" ]; then
    echo "⚠️  .env file not found in project root. Creating template..."
    cat > ../../../../.env << 'EOF'
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration (OpenAI or compatible)
LLM_API_KEY=your_openai_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_CHOICE=gpt-4o-mini

# Embedding Configuration
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSION=1536

# Intercom Configuration (optional - will use mock data if not provided)
INTERCOM_ACCESS_TOKEN=your_intercom_token_here
EOF
    echo "✅ Created .env template. Please edit it with your actual values."
    echo "Required variables: NEO4J_PASSWORD, LLM_API_KEY, EMBEDDING_API_KEY"
    echo "Optional: INTERCOM_ACCESS_TOKEN (mock data will be used if not provided)"
    echo ""
    read -p "Press Enter to continue with the test or Ctrl+C to exit and configure .env first..."
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../../.."

# Run the test
echo "🏃 Running the test..."
python test_ingest_kg.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
else
    echo ""
    echo "❌ Test failed. Check the logs above for details."
    exit 1
fi 