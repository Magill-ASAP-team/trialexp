from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated data source
data = {
    "x": list(range(100)),
    "y": [i**0.5 for i in range(100)],
}

@app.get("/")
async def get_root():
    return HTMLResponse("Backend for Signal Dashboard")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            print('receved ws message')
            message = await websocket.receive_json()
            if "zoom" in message:
                # Simulate updated data based on zoom
                zoom_range = message["zoom"]
                print(zoom_range)
                updated_data = {
                    "x": list(range(zoom_range[0], zoom_range[1] + 1)),
                    "y": [i**0.5 for i in range(zoom_range[0], zoom_range[1] + 1)],
                }
                await websocket.send_json(updated_data)
    except Exception as e:
        print(f"Connection closed: {e}")
