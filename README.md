FPLDraftDash
===========

Simple Dash dashboard for FPL Draft 2025/26.

Prerequisites
-------------
- Python 3.12
- Recommended: use a virtual environment

Quick start (PowerShell)
------------------------
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python FPLDraftDash.py
```

If you already have a project-specific Python executable (configured in this workspace), run it instead, for example:

```powershell
c:/Python/venv/py312/Scripts/python.exe FPLDraftDash.py
```

Notes
-----
- `requirements.txt` contains the minimal runtime dependencies used by the app.
- `pyproject.toml` contains project metadata and the same dependencies for packaging or tools that read PEP 621 metadata.
