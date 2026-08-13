# Support Assistant Submodule Documentation

This service runs an automated customer support routing engine driven by LangGraph state routing loops and an embedded ChromaDB knowledge base.

## Ingestion, Embedding, and RAG Pipeline Architecture Description

The document extraction process follows four linear processing phases:

```text
 [ Ingestion ] ──> [ Embedding ] ──> [ Retrieval ] ──> [ Generation ]
 (seed_docs.py)   (all-MiniLM-L6-v2)   (ChromaDB Query)   (MOCK_LLM Switch)
```

* **Ingestion (`seed_docs.py`):** Collects the 8 raw text corporate policy guidelines and writes them to disk under the `/docs` directory.
* **Embedding (`main.py`):** Processes the text blocks into dense numeric vectors using a local `sentence-transformers/all-MiniLM-L6-v2` engine. These vectors are indexed on-disk inside a persistent `ChromaDB` collection.
* **Retrieval (`classify_intent` & `retrieve_and_answer` nodes):** User lookups are received by the `classify_intent` node. If a keyword match is found, the state engine uses a conditional edge to route the query to `retrieve_and_answer`. This node runs a cosine similarity lookup against ChromaDB to extract the top-3 matching document chunks.
* **Generation (`MOCK_LLM` execution toggles):**
  - **`MOCK_LLM=1` (Default Graded Baseline):** Bypasses all network overhead. The system uses local data to return a structured response containing a canned string built around the top text snippet.
  - **`MOCK_LLM=0` (Live LLM Extension):** Passes the extracted text context directly into a real LLM using our structured role-context-task template. It includes a 2-stage corrective retry loop to fix any malformed JSON outputs.

---

## Local Container Orchestration Validation

To build and test the microservice locally using Docker, run these commands in your terminal:

```bash
cd support_assistant
# Compile the container image layers
docker build -t zepto-support-assistant:latest -f Dockerfile ..

# Run the container background thread locally
docker run -d -p 7860:7860 --name zepto-assistant-node zepto-support-assistant:latest
```

Open `http://127.0.0` in your browser to view the interactive documentation and test endpoints using the Swagger UI.

---

## Logged Web Service API Integration Transcripts (`MOCK_LLM=1`)

These logs verify that our graph intent router and Pydantic validation schemas operate correctly under our standard mock testing parameters:

### Test Case 1: In-Scope Policy Query (Triggers Document Retrieval Path)
* **Endpoint Called:** `POST http://127.0.0`
* **Request JSON:**
```json
{
  "query": "Can I cancel my grocery order if it hasn't been packed yet?"
}
```
* **Response JSON:**
```json
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can n...",
  "sources": [
    "doc_05"
  ],
  "confidence": 1.0
}
```

### Test Case 2: Out-of-Scope General Query (Bypasses Retrieval Path)
* **Endpoint Called:** `POST http://127.0.0`
* **Request JSON:**
```json
{
  "query": "What is the average surface temperature profile of Mars?"
}
```
* **Response JSON:**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
