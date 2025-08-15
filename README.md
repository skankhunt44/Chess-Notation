To run use 2 terminal

terminal 1:
uvicorn app:app --reload --host 0.0.0.0 --port 8000

uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75


terminal 2:
ngrok http 8000

Open `http://localhost:8000` in a browser. After granting camera access press
the **Calibrate board** button once with an empty board visible. Set up the
pieces and press the button again to start tracking.

The server keeps track of all recognised moves.  Fetch the current move
history (in SAN) via:

```
GET /history
```
