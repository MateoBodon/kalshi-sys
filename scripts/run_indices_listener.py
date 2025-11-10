import asyncio
import json
import websockets

from kalshi_alpha.utils.keys import load_polygon_api_key

CHANNEL = "A.I:SPX,A.I:NDX"
URI = "wss://socket.massive.com/indices"


async def _stream_forever(timeout: float = 5.0) -> None:
    api_key = load_polygon_api_key()
    while True:
        try:
            async with websockets.connect(
                URI,
                ping_interval=60,
                ping_timeout=60,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({"action": "auth", "params": api_key}))
                print("AUTH:", await ws.recv())
                await ws.send(json.dumps({"action": "subscribe", "params": CHANNEL}))
                print("SUB:", await ws.recv())
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        print("MSG:", msg)
                    except asyncio.TimeoutError:
                        print("TIMEOUT: no data, keeping connection alive")
        except websockets.ConnectionClosed as exc:
            print(f"CLOSED: {exc.code} {exc.reason}")
            await asyncio.sleep(1.0)
        except Exception as exc:
            print(f"ERROR: {exc}")
            await asyncio.sleep(1.0)


def main() -> None:
    asyncio.run(_stream_forever())


if __name__ == "__main__":
    main()
