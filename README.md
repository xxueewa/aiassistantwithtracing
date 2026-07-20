## BizTrip Voice Assistant

<img width="800" height="450" alt="biztrip-product-showcase" src="https://github.com/user-attachments/assets/dffb029f-d8a4-4d97-8910-450d65ce7a01" />
<br>

|LangGraph|LangSmith|AWS|iOS|
|---|---|---|---|
|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/d1183de1-1b14-4207-b129-fb7d87fc48e4" />|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/e85f784a-861c-47c3-a26e-98964706e18d" />|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/80aed474-9e90-479b-9f8d-be7b01606405" />|<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/eae87b04-b1c7-4f23-96da-cc25084607f6" />|

<br>
A voice-first travel companion of staff for business trips: it knows your itinerary, company policy, meetings, expenses, and local culture, and can take actions while you’re moving. 

***Contextual recall:97.5% | Precision: 67.56% | Tool Selection Accuracy: 88% | Task Correctness: 80%***

### Features
- Audio & Text Interaction on Mobile App
- Intention Detection and Self-Reasoning
- Enterprise Knowledge Base
- Online Search
- System Tracing and Performance Monitor 

### Coding Agent - Codex & GPT-5.6
- Audio Processing
- Product Document
- API Contract Review

### Development
- run services: in each module, execute `uv run uvicorn main:app --reload`
- run langgraph agent: `langgraph dev`

### Product Description

Business travelers often move quickly between airports, train stations, hotels, and meeting locations, where typing questions or manually searching for information is inconvenient. BizTrip Voice Assistant was inspired by the need for a hands-free assistant that can understand travel-related requests, retrieve enterprise knowledge, and help users complete tasks efficiently while they are on the move.

BizTrip Voice Assistant listens to user requests, understands both speech and text input, and resolves tasks using company policies, project documents, travel preferences, and organizational knowledge. It remembers user preferences, supports personalized communication, automates repetitive business travel tasks, and helps users stay focused during the trip.

Bring BizTrip. Enjoy your business travel.

BizTrip is built with LangGraph Agent Harness, Retrieval-Augmented Generation, Voice Activity Detection, and Kafka-based event processing.

Key challenges included:
1. Handling high-quality audio streaming input, since audio format, sample rate, and data integrity directly affect transcription accuracy and overall task correctness.
2. Designing an efficient RAG indexing and retrieval pipeline to balance correctness, relevance, and low latency.
3. Structuring the agent graph around business travel subtasks, including planning, in-trip assistance, reimbursement, and follow-up management.
4. Evaluating agent behavior across tool selection, retrieval quality, and final answer correctness.