To run use 2 terminal

terminal 1:
uvicorn app:app --reload --host 0.0.0.0 --port 8000

terminal 2:
ngrok http 8000

The server keeps track of all recognised moves.  Fetch the current move
history (in SAN) via:

```
GET /history
```
