# SVG to Lottie Converter

This repository provides two ways to convert SVG files to Lottie JSON:

- A FastAPI-based Web service (`src/svgtolottie.py`) with an upload endpoint `/uploadsvg/`.
- A command-line interface (CLI) tool (`src/cli.py`) that can be invoked as `python src/cli.py input.svg output.json`.

Both approaches use the same conversion logic under `src/core/svg/convert.py` and `src/model/` Pydantic models.

---

## Requirements
```bash
pip install -r requirements.txt
```

Note: This project was developed with Python 3.8/3.11 (3.8+ recommended). If you face compatibility issues (e.g., `pydantic.Schema` import errors), try using the virtual environment setup below.

---

## Quick Setup (recommended)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 1) CLI Usage (recommended for single conversions)

You can use the CLI to convert a local SVG file to Lottie JSON with a single command.

Basic usage:
```bash
python src/cli.py input.svg output.json
```

Options:
- `--optimize`: use the optimized conversion flow
- `--compact`: produce compact JSON (no indentation)
- `-o`, `--output-file`: alternative to positional output argument
- `-q`, `--quiet`: suppress status output

Examples:
```bash
# Convert and save pretty JSON
python src/cli.py src/file_copy.svg output/cli_output.json

# Convert with compact JSON
python src/cli.py src/file_copy.svg output/compact.json --compact

# Convert with the optimized flow
python src/cli.py src/file_copy.svg output/optimized.json --optimize
```

The CLI will print conversion stats and save the result to the specified path. If no output filename is provided, it will save to `<input>.json`.

---

## 2) Web API Usage (upload from browser/remote client)

Start the FastAPI server:
```bash
python src/svgtolottie.py
# Server will be accessible at http://0.0.0.0:8000
```

Upload endpoint:

- `POST /uploadsvg/` - accepts an `UploadFile` named `file` and optional query params:
  - `optimize`: boolean flag to use optimized flow
  - `output_path`: optional path to save the generated JSON on the server

Example usage with curl (returns JSON in response):
```bash
curl -X POST "http://127.0.0.1:8000/uploadsvg/" -F "file=@file_copy.svg"
```

Example usage to save on the server:
```bash
curl -X POST "http://127.0.0.1:8000/uploadsvg/?output_path=output/my_lottie.json" -F "file=@file_copy.svg"
```

When `output_path` is provided, the API will save the JSON to the specified location on disk and return the same JSON along with a `message`, `output_path`, and `data` fields.

---

## Files of Interest
- `src/svgtolottie.py` - the FastAPI app entry point
- `src/cli.py` - the command-line interface wrapper
- `src/core/svg/convert.py` - conversion implementation
- `src/core/svg/gradients.py` - color parsing and gradient handling
- `src/model/` - Pydantic models used to structure the Lottie JSON

---

## Troubleshooting
- If you see `ImportError: cannot import name 'Schema' from 'pydantic'`, use the updated `requirements.txt` and ensure `pydantic` is < 2.0 (the repository already updated code to remove `Schema` imports).
- If you see `ModuleNotFoundError: No module named 'triangle'`, install the `triangle` package (`pip install triangle`) or use `requirements.txt`.
- If you see color parse `KeyError` from `rgb(...)`, the regex has been updated to support floating percent values.
- If you see `ERROR:    [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use`, ensure no other process is using port 8000 or change the port in `src/svgtolottie.py`. You can use `lsof -i :8000` and `kill $(cat logs/uvicorn.pid)` to find the process using the port and kill it if necessary.

---

If you'd like, I can add unit tests and CI configuration to validate CLI and API runs. 
