# Document QA Chatbot

Streamlit + LangChain + Gemini app that answers questions about an uploaded PDF (RAG).

## Run locally with Docker
```bash
cp .env.example .env        # then put your real GOOGLE_API_KEY in .env
docker compose up --build   # open http://localhost:8501
```

## Run without Docker
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| GOOGLE_API_KEY | required | Google AI API key |
| CHAT_MODEL | gemini-4-hydrogen | Chat model name |
| EMBED_MODEL | gemini-embedding-2-preview | Embedding model name |
| TOP_K | 4 | Chunks retrieved per question |
| LOG_LEVEL | INFO | Logging level |

## Git workflow
- `main` is protected; work on feature branches and open pull requests.
- CI lints and builds on every PR; merging to `main` publishes the image to GHCR.
- Never commit `.env` or any PDF with patient data.

## Production checklist
- Inject `GOOGLE_API_KEY` from your platform's secret store, not from a file in the image.
- Put the app behind HTTPS (reverse proxy or load balancer) and add authentication.
- Pin exact versions in `requirements.txt` (`pip freeze`) for reproducible builds.
- Data is held in memory only and lost on restart.
