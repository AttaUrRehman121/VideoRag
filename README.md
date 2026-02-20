# Video RAG (Simplified) - Django Backend

This is a simplified Django REST backend for asking questions about video transcripts using RAG.

## What’s included

- SQLite database (`db.sqlite3`)
- Transcript ingestion (chunk + embed + store)
- Retrieval with cosine similarity (Python-side)
- Answer generation via OpenAI
- Question/answer caching (`QuestionCache`)

## Setup

1) Create `.env` from the example:

```bash
copy .env.example .env
```

2) Set at least:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
SECRET_KEY=change-me
```

3) Install and migrate:

```bash
pip install -r requirements.txt
python manage.py migrate
```

4) Run:

```bash
python manage.py runserver
```

## API

- `POST /api/ai/chunks/ingest/`
- `POST /api/ai/ask/ask/`

### Ingest transcript

```bash
curl -X POST http://127.0.0.1:8000/api/ai/chunks/ingest/ ^
  -H "Content-Type: application/json" ^
  -d "{\"video_id\":\"1234\",\"video_title\":\"Optional title\",\"transcript\":\"Full transcript text...\"}"
```

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/api/ai/ask/ask/ ^
  -H "Content-Type: application/json" ^
  -d "{\"video_id\":\"1234\",\"question\":\"How many roles are in this system?\"}"
```

## Notes

- Embeddings are stored as JSON text in SQLite; retrieval is Python-side cosine similarity.
- Cache entries are saved after responding (background thread).

