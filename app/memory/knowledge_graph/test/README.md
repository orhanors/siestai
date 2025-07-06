# Knowledge Graph Ingestion Test

This test script demonstrates how to ingest data from Intercom into a Neo4j knowledge graph using the KGClient and Graphiti.

## What This Test Does

1. **Connects to Neo4j** - Uses the local Neo4j instance (configured in docker-compose.yaml)
2. **Fetches Intercom Data** - Retrieves articles from Intercom API (or uses mock data if not configured)
3. **Ingests into Knowledge Graph** - Adds documents as episodes to the Graphiti knowledge graph
4. **Tests Search Functionality** - Performs semantic searches on the ingested data
5. **Tests Entity Relationships** - Queries for entity relationships and facts

## Prerequisites

### 1. Start Neo4j Database

Make sure Neo4j is running locally. From the project root:

```bash
docker-compose up -d neo4j
```

The Neo4j instance will be available at:
- Web interface: http://localhost:7474
- Bolt connection: bolt://localhost:7687
- Default credentials: neo4j/your_password

### 2. Install Dependencies

Make sure you have all required Python dependencies installed:

```bash
pip install -r requirements.txt
# or if using poetry:
poetry install
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with these variables:

```env
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
```

**Required Variables:**
- `NEO4J_PASSWORD` - Your Neo4j password
- `LLM_API_KEY` - OpenAI API key for LLM operations
- `EMBEDDING_API_KEY` - OpenAI API key for embeddings (can be the same as LLM_API_KEY)

**Optional Variables:**
- `INTERCOM_ACCESS_TOKEN` - If not provided, the test will use mock data

## Running the Test

### Option 1: Using the Shell Script (Recommended)

```bash
cd app/memory/knowledge_graph/test
./run_test.sh
```

The shell script will:
- Check if Neo4j is running
- Verify environment variables
- Create a .env template if needed
- Run the test with proper Python path setup

### Option 2: Running Python Script Directly

```bash
cd app/memory/knowledge_graph/test
python test_ingest_kg.py
```

Make sure your `PYTHONPATH` includes the project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../../.."
```

## Test Output

The test will show progress with colored emojis:
- ✅ Success
- ❌ Error
- ⚠️ Warning
- 🚀 Starting
- 🔍 Checking

Example output:
```
🚀 Starting full KG ingestion test...
Setting up test environment...
✅ KG connection successful. Stats: {'graphiti_initialized': True, 'sample_search_results': 0}
✅ Fetched 3 documents from Intercom
✅ Ingested episode 1/3: Getting Started with Customer Support...
✅ Found 2 results for 'customer support'
✅ Full test completed successfully!
```

## What Gets Ingested

The test ingests Intercom articles as "episodes" in the knowledge graph:
- **Episode ID**: `intercom_article_{original_id}_{index}`
- **Content**: Combined title and article content
- **Source**: Intercom article source with URL
- **Metadata**: Article tags, author, state, etc.
- **Timestamp**: Article creation date

## Troubleshooting

### Neo4j Connection Issues
- Ensure Neo4j container is running: `docker ps | grep neo4j`
- Check Neo4j password matches your .env file
- Verify port 7687 is not blocked

### LLM/Embedding Issues
- Verify OpenAI API key is valid
- Check API rate limits
- Ensure you have sufficient OpenAI credits

### Intercom Issues
- If no Intercom token is provided, mock data will be used
- Verify Intercom token has read permissions
- Check Intercom API rate limits

### Python Import Issues
- Ensure you're running from the correct directory
- Check PYTHONPATH includes project root
- Verify all dependencies are installed

## Advanced Usage

### Customizing the Test

You can modify the test parameters in `test_ingest_kg.py`:

```python
# Change number of documents to fetch
documents = await self.fetch_intercom_data(limit=10)

# Modify search queries
search_queries = [
    "your custom query",
    "another query"
]

# Change entity relationships to test
entities = ["your", "entities", "here"]
```

### Using Real Intercom Data

To use real Intercom data instead of mock data:

1. Get an Intercom access token from your Intercom app settings
2. Add it to your .env file: `INTERCOM_ACCESS_TOKEN=your_token_here`
3. Run the test - it will automatically use real data

### Clearing the Knowledge Graph

The test includes a method to clear all data from the knowledge graph:

```python
await self.kg_client.clear_graph()
```

**⚠️ Warning**: This will permanently delete all data in your Neo4j database!

## Next Steps

After running this test successfully, you can:
1. Explore the Neo4j web interface to see the ingested data
2. Modify the script to ingest different types of data
3. Experiment with different search queries
4. Build upon this foundation for your production data ingestion pipeline 