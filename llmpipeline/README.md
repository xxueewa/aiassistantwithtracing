### AI Assistant Project with Tracing System

### Introduction
This project aims to design and build an LLM system with essential building blocks. In user point
of view, it starts with supporting the basic llm capabilities, such as answering questions based on domain knowledge. Increasingly, I will 
extend its functionalities to function calling and reasoning. In the aspect of system design, it covers
tracing service and evaluation service. The tracing service is designed to log the execution of each sub-task. 
It ensures the proper execution of each step, and would be used to troubleshoot complex tasks's flow. 
The evaluation service is designed to enhance the system's performance after the basic functionalities are completed.

### High-level Architecture
<img width="3326" height="2442" alt="High-Level-Design (langchain project))" src="https://github.com/user-attachments/assets/ab4ed659-ce29-4fd7-abb3-b68a1de504b0" />


### Key Component
1. Retrieval Augmented Generation (RAG) Service
   - Content includes: domain knowledge ()
   - Storage: ver1. In-Memory Vector Store; ver2. AWS OpenSearch Service 
   - Embedding model: text-embedding-3-small (dev)
   - Search methods: similarity search, similarity search with score, similarity search with vector
2. Tracing and Monitoring Service 
   - LangSmith 
   - Log content: Document upsert (result and size), Document retrieval (search method, score/ranking), LLM response  
   - Metrics: 
     - Trace latency (P50, P99)
     - Trace error rate
     - Cost over time
     - Output/Input tokens
     - Tool counts over time
     - Tool latency
     - Tool error rate
3. Evaluation Service
   - Reference-free Evaluator
     - Safety check: Toxicity detection, PII detection, Content policy violation
     - Format validation: JSON, required fields, schema compliance
     - Quality heuristics: Response length, latency, specific keywords
     - Reference-free LLM-as-judge: clarity, coherence, helpfulness, tone
   - Reference-based Evaluator
     - Correctness: Semantic similarity to reference answer
     - Factual accuracy: Fact-checking against ground truth
     - Exact match: classify tasks with known labels
     - Reference-based LLM-as-judge: comparing output quality to a reference answer

### Milestone
- [x] 2026-04-16: Completed project basic functions and set up the tracing: https://shidemuxue.atlassian.net/wiki/x/AQAo
- [] 2026-04-xx: select the search method and evaluate the retrieval performance
- [] 2026-04-xx: 