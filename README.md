# Chess-Notation

## Installation

Install Python dependencies:

```
pip install -r requirement.txt
```

## Running the server

Start the FastAPI application:

```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` automatically reloads the server on code changes and can be
omitted in production.

To make the server accessible from the internet (optional), run in another
terminal:

```
ngrok http 8000
```

## Usage

Open `http://localhost:8000` in a browser (or the URL provided by ngrok).
After granting camera access:

1. Press the **Calibrate board** button once with an empty board visible.
2. Set up the pieces.
3. Press **Calibrate board** again to start tracking.

The server keeps track of all recognised moves. Fetch the current move
history (in SAN) via:

```
GET /history
```

