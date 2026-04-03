# Prereq Chatbot Workspace
## run evals 
-$env:PYTHONPATH="$PWD\apps\api\src"
-set RAG_PIPELINE_DELAY_SECONDS=45
-uv run --env-file .env python apps/api/evals/eval_retriever.py
A small multi-app workspace for running a chatbot with:

- a FastAPI backend (`apps/api`)
- a Streamlit frontend (`apps/chatbot-ui`)
- provider support for Groq and Gemini models

## Tech Stack

- Python 3.12+
- `uv` for dependency/workspace management
- FastAPI + Uvicorn (API)
- Streamlit (UI)
- Docker + Docker Compose (optional, recommended)

## Project Structure

- `apps/api`: FastAPI chat API (`POST /chat`)
- `apps/chatbot-ui`: Streamlit chat interface
- `docker-compose.yaml`: Runs both services together
- `Makefile`: Convenience commands

## Prerequisites

Install:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop (if using containers)

## Environment Variables

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
# OR use GOOGLE_API_KEY instead of GEMINI_API_KEY
# GOOGLE_API_KEY=your_google_api_key
HF_API_TOKEN=your_huggingface_token_optional
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://qdrant:6333
```

Notes:

- `GROQ_API_KEY` is required for Groq provider.
- For Gemini, set either `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- `HF_API_TOKEN` is optional but recommended for higher Hugging Face API limits.
- `HF_EMBEDDING_MODEL` configures the remote embedding model used by the RAG pipeline.
- `QDRANT_URL` should be `http://qdrant:6333` in Docker Compose, `http://localhost:6333` if running API locally.
- `.env` is gitignored in this repo.

## Run with Docker Compose

From the project root:

```bash
uv sync
docker compose up --build
```

Or with Make:

```bash
make run-docker-compose
```

Services:

- UI: `http://localhost:8501`
- API: `http://localhost:8000`

## Run Locally (Without Docker)

1. Install dependencies:

```bash
uv sync
```

2. Start API (terminal 1):

```bash
uv run --package api uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

3. Start UI (terminal 2):

```bash
uv run --package chatbot-ui streamlit run apps/chatbot-ui/src/chatbot_ui/app.py
```

4. Open `http://localhost:8501`.

If running locally, the UI may need `API_URL` set to `http://localhost:8000` in `.env` (instead of the Docker default `http://api:8000`).

## API Endpoint

- `POST /chat`

Request body:

```json
{
  "provider": "groq",
  "model_name": "llama-3.3-70b-versatile",
  "messages": [
    { "role": "user", "content": "Hello" }
  ]
}
```

Response:

```json
{
  "message": "..."
}
```

## Handy Commands

- `make run-docker-compose`: install deps and start compose
- `make clean-notebook-outputs`: clear outputs in `notebooks/*/*.ipynb`

