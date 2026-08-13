# zepto-aiml-capstone
Capstone project 
# Zepto Data Pipeline, Analytics, and Support Assistant Workspace

An integrated three-tier repository containing automated data pipelines, machine learning analytics suites, and AI customer support assistants.

## System Dependencies and Configuration

This workspace utilizes a **consolidated requirements tracking system** located at the root level. Install all project constraints simultaneously by running:

```bash
pip install -r requirements.txt
```

---

## Architectural Layout & System Orchestration

### 1. Data Pipeline Module (`/data_pipeline`)
* **Execution:** Run `python data_pipeline/pipeline.py`.
* **Core Function:** Automated scraping engine targeting books catalog records. Normalizes pricing to standard currency baselines (**1 GBP = 105.50 INR**), structures records into a strict primary/foreign key SQLite instance, and cross-validates engine integrity with pandas mapping functions.

### 2. Analytics Module (`/analytics`)
* **Execution:** Open VS Code and run `01_eda.ipynb` and `02_modeling.ipynb` sequentially.
* **Core Function:** Handles exploratory profiling, data cleaning, and class-imbalance treatments on the Titanic dataset before saving a serialized production pipeline model artifact (`titanic_production_pipeline.joblib`).

### 3. Support Assistant Module (`/support_assistant`)
* **Execution:** Navigate to the folder, seed the docs using `python seed_docs.py`, and start the server using `python main.py`.
* **Core Function:** Local RAG framework utilizing `ChromaDB` and `all-MiniLM-L6-v2` embeddings to deliver grounded corporate policy solutions. Responses are strictly validated against structured JSON data schemas.

---

## Workspace Version History

This repository manages code changes using a structured branch strategy. Functional iterations are developed in a dedicated feature branch, verified through sequential commits, and subsequently integrated cleanly into the primary `main` branch.
