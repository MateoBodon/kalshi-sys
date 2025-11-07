import asyncio
import json

import websockets

from kalshi_alpha.utils.keys import load_polygon_api_key


async def main() -> None:
    api_key = load_polygon_api_key()
    uri = "wss://socket.massive.com/indices"
    ws: websockets.WebSocketClientProtocol | None = None
    try:
        ws = await websockets.connect(
            uri,
            ping_interval=60,
            ping_timeout=60,
            close_timeout=5,
        )
        await ws.send(json.dumps({"action": "auth", "params": api_key}))
        print("AUTH:", await ws.recv())
        await ws.send(json.dumps({"action": "subscribe", "params": "A.I:SPX,A.I:NDX"}))
        print("SUB:", await ws.recv())
        try:
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print("MSG:", msg)
        except websockets.ConnectionClosed as exc:
            print("CLOSED:", exc.code, exc.reason)
        except asyncio.TimeoutError:
            print("TIMEOUT: no data in 5s, closing...")
    finally:
        if ws is not None and not ws.closed:
            await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
