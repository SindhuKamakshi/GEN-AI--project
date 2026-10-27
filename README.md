# CodeLens AI

CodeLens AI is a static HTML/CSS/JavaScript frontend with a FastAPI backend for
repository indexing, hybrid code search, evidence-backed questions, and
Python/notebook dependency analysis.

## Local development

Use Python 3.14 or another Python version supported by the pinned packages.
Install backend dependencies once:

```powershell
cd C:\Users\sindh\OneDrive\Desktop\CodeLens-AI\backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in the project-root `.env` file. Never put provider keys in
frontend files. Optional model settings are `GEMINI_PRIMARY_MODEL` and
`GEMINI_FALLBACK_MODEL`; the defaults are `gemini-3.8-flash` and
`gemini-3.5-flash-lite`.

Start the backend in one PowerShell terminal:

```powershell
cd C:\Users\sindh\OneDrive\Desktop\CodeLens-AI\backend
.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Serve the frontend in a second terminal:

```powershell
python -m http.server 5500 --bind 127.0.0.1 --directory C:\Users\sindh\OneDrive\Desktop\CodeLens-AI\frontend
```

Open <http://127.0.0.1:5500/>.

Run backend tests from the backend directory:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

## Deployment configuration

The frontend is static. Set the `codelens-api-base` meta tag in
`frontend/index.html` to the deployed backend origin (for example,
`https://api.example.com`). The backend reads these server-side environment
variables:

- `GEMINI_API_KEY` (required for generated answers)
- `GEMINI_PRIMARY_MODEL` and `GEMINI_FALLBACK_MODEL` (optional)
- `CODELENS_ALLOWED_ORIGINS` (comma-separated exact frontend origins)
- `CODELENS_DATA_DIR` (optional path for repository uploads, ChromaDB, and the
  SQLite keyword index)

Mount `CODELENS_DATA_DIR` on durable storage before deploying. Repository
uploads, ChromaDB, and the SQLite FTS index are local files; they will not
survive an ephemeral instance restart unless the hosting platform provides a
persistent volume. Keep the backend on one instance unless the local indexes
are migrated to shared services with appropriate concurrency controls.

For a production backend, run Uvicorn bound to `0.0.0.0` and the port supplied
by the hosting platform. Set `CODELENS_ALLOWED_ORIGINS` to the exact HTTPS
frontend origin, `CODELENS_DATA_DIR` to the mounted persistent-volume path, and
`GEMINI_API_KEY` in the platform's secret manager. Configure the frontend API
meta tag to the public backend HTTPS origin. Do not use the development
`--reload` option in production.

No hosting platform has been selected or deployed from this workspace. The
platform's persistent-volume policy, resource limits, and public URLs must be
confirmed before deployment.