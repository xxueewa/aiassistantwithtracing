## AI Assistant Application

|LangGraph|LangSmith|
|---|---|
|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/d1183de1-1b14-4207-b129-fb7d87fc48e4" />|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/e85f784a-861c-47c3-a26e-98964706e18d" />|

<br>
Built an intelligent assistant that utilized LLM Mixture-of-Experts for sharp reasoning and context awareness — augmented with function calling and RAG to unlock specialized domain knowledge.
Leveraged LangSmith for comprehensive tracing, monitoring, and observability across the full agent lifecycle.

### Features
- Audio & Text Interaction on Mobile App
- Intention Detection and Self-Reasoning
- Incremented Domain Knowledge from Chat History
- Admin Console to Manage User Approved Domain Knowledge 
- System Tracing and Performance Monitor 

### Development
- run services: in each module, execute `uv run uvicorn main:app --reload`
- run langgraph agent: `langgraph dev`
