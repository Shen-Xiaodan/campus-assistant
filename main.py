"""Compatibility entry point. Prefer `uvicorn app.api:app` and `streamlit run app/ui.py`."""

from app.api import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
