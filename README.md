uvicorn app:app --host 0.0.0.0 --port 8000 --reload
uv run --no-sync --with "numpy<2" fastapi run app.py