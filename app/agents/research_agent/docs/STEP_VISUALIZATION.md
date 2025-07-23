# LangGraph Step Visualization

This document describes the step-by-step visualization system implemented for the Research Agent using LangGraph features.

## 🎯 Features Implemented

### 1. **Real-time Step Tracking**
- Each LangGraph workflow step is tracked in real-time
- Shows step start/completion with timestamps
- Displays step-specific results and metrics
- Tracks execution time for performance analysis

### 2. **Detailed Step Information**
Each step provides specific insights:

#### `retrieve_memory`
- Memory context found (yes/no)
- Number of similar messages from history
- Number of recent context messages
- Timing: ~0.5-1.0s

#### `retrieve_documents` 
- Number of documents found
- Document titles and sources
- Vector search results
- Timing: ~0.4-0.6s

#### `search_knowledge_graph`
- Number of KG entities found
- Entity names and descriptions
- Graph traversal results
- Timing: ~0.0s (disabled in current setup)

#### `web_search`
- Number of web results retrieved
- Web page titles and URLs
- Tavily search integration
- Timing: ~3-6s (depends on network)

#### `synthesize_answer`
- Generated response length
- Sources used in synthesis
- LLM processing results
- Timing: ~1-6s (depends on complexity)

### 3. **Performance Metrics**
- Individual step timing
- Total execution time
- Step percentage breakdown
- Performance bottleneck identification

### 4. **Interactive Display**
- Rich terminal formatting with colors
- Real-time progress updates
- Detailed summary tables
- Memory context visualization

## 🔧 Implementation Details

### Core Components

#### 1. **Step Tracking Decorator**
```python
@track_step("step_name")
async def step_function(self, state: ResearchState) -> ResearchState:
    # Step implementation
    pass
```

#### 2. **Enhanced State Management**
```python
class ResearchState(TypedDict):
    # ... existing fields ...
    current_step: str
    step_results: Dict[str, Any]
    step_timings: Dict[str, float]
```

#### 3. **Callback System**
```python
agent = ResearchAgent(
    # ... other parameters ...
    step_callback=tracker.step_callback
)
```

### LangGraph Workflow Integration
The visualization integrates seamlessly with the LangGraph workflow:

```
START → retrieve_memory → retrieve_documents → search_kg → [web_search] → synthesize → END
  ↓           ↓                ↓                ↓              ↓           ↓
Track      Track           Track           Track         Track       Track
```

## 📊 Usage Examples

### 1. **Basic Step Tracking**
```python
from app.agents.research_agent import ResearchAgent

async def step_callback(message: str, state):
    print(f"Step: {message}")

agent = ResearchAgent(step_callback=step_callback)
result = await agent.research("What is AI?")

# Access step information
print(result["step_timings"])
print(result["step_results"])
```

### 2. **Advanced Visualization**
See `test/test_steps.py` for a complete example with:
- Rich terminal output
- Detailed step summaries
- Performance analysis
- Memory tracking

### 3. **Chat Interface Integration**
The main chat interface (`test/chat.py`) includes:
- Real-time step progress display
- Step summary table after each query
- Memory context visualization
- Performance metrics

## 🎪 Demo Scripts

1. **`test/test_steps.py`**: Full-featured demo with detailed step visualization
2. **`test/quick_test.py`**: Simple test showing basic step tracking
3. **`test/chat.py`**: Interactive chat with integrated step display

## 🔍 Step-by-Step Example Output

```
🔄 Starting: retrieve_memory
✅ Completed: retrieve_memory (0.84s)
   🧠 Memory: 2 similar, 3 recent messages

🔄 Starting: retrieve_documents  
✅ Completed: retrieve_documents (0.54s)
   📄 Documents: Found 2 relevant documents
      1. LangGraph Framework Guide
      2. Internal LangGraph Guide

🔄 Starting: web_search
✅ Completed: web_search (3.65s)
   🌐 Web Search: Found 2 results
      1. How Does AI Work? - Coursera
      2. What is AI? - Tableau

🔄 Starting: synthesize_answer
✅ Completed: synthesize_answer (1.41s)
   🔮 Synthesis: Generated 843 character response
   📊 Sources used: documents, web_search

⏱️  TIMING SUMMARY:
   Total time: 6.44s
   retrieve_memory: 0.84s (13.0%)
   retrieve_documents: 0.54s (8.4%)
   web_search: 3.65s (56.7%)
   synthesize_answer: 1.41s (21.9%)
```

## 🚀 Benefits

1. **Transparency**: See exactly what the agent is doing at each step
2. **Performance**: Identify bottlenecks and optimize slow steps
3. **Debugging**: Quickly identify where issues occur
4. **User Experience**: Users understand the research process
5. **Memory Insights**: See how chat history influences responses
6. **Development**: Easier to tune and improve individual steps

## 🔄 Integration with Memory System

The step visualization works seamlessly with the chat history system:
- Shows memory retrieval progress
- Displays similar messages found
- Tracks memory context usage
- Visualizes user/profile isolation

This creates a complete picture of how the agent processes queries using both current context and historical memory.