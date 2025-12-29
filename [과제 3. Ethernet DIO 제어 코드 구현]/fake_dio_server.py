# fake_dio_server.py

import asyncio
from typing import Tuple

HEADER = 0xAA
RESERVED = 0x00

FT_GET_INPUT = 0xC0
FT_GET_OUTPUT = 0xC5
FT_SET_OUTPUT = 0xC6

COMM_OK = 0x00
COMM_ERR = 0x01


def build_resp(sync: int, frame_type: int, comm: int, reply_data: bytes) -> bytes:
    body = bytes([sync & 0xFF, RESERVED, frame_type & 0xFF, comm & 0xFF]) + reply_data
    length = len(body)
    return bytes([HEADER, length & 0xFF]) + body


def parse_req(packet: bytes) -> Tuple[int, int, bytes]:
    if len(packet) < 5:
        raise ValueError("req too short")
    if packet[0] != HEADER:
        raise ValueError("bad header")
    length = packet[1]
    if len(packet) != 2 + length:
        raise ValueError("length mismatch")
    sync = packet[2]
    reserved = packet[3]
    ft = packet[4]
    if reserved != RESERVED:
        raise ValueError("reserved mismatch")
    data = packet[5:]
    return sync, ft, data


class FakeDioDevice:
    """
    - input_mask32: bit0~7 사용
    - output_mask32: bit8~15 사용 (I8O8처럼)
    """
    def __init__(self) -> None:
        self.input_mask32 = 0
        self.output_mask32 = 0

    def handle(self, sync: int, ft: int, data: bytes) -> bytes:
        if ft == FT_GET_INPUT:
            # reply_data: Input(4) + Latch(4)  (총 8바이트)
            reply = self.input_mask32.to_bytes(4, "little") + (0).to_bytes(4, "little")
            return build_resp(sync, ft, COMM_OK, reply)

        if ft == FT_GET_OUTPUT:
            # reply_data: Output(4) + Run/Stop(4) (총 8바이트)
            reply = self.output_mask32.to_bytes(4, "little") + (0).to_bytes(4, "little")
            return build_resp(sync, ft, COMM_OK, reply)

        if ft == FT_SET_OUTPUT:
            if len(data) != 8:
                return build_resp(sync, ft, COMM_ERR, b"")
            set_mask = int.from_bytes(data[0:4], "little")
            reset_mask = int.from_bytes(data[4:8], "little")

            # reset -> set 순으로 적용
            self.output_mask32 &= ~reset_mask
            self.output_mask32 |= set_mask

            return build_resp(sync, ft, COMM_OK, b"")

        return build_resp(sync, ft, COMM_ERR, b"")


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, dev: FakeDioDevice):
    addr = writer.get_extra_info("peername")
    print(f"[FAKE] client connected: {addr}")
    try:
        while True:
            header = await reader.readexactly(2)
            if header[0] != HEADER:
                print("[FAKE] bad header -> close")
                return
            length = header[1]
            body = await reader.readexactly(length)
            packet = header + body

            try:
                sync, ft, data = parse_req(packet)
                resp = dev.handle(sync, ft, data)
            except Exception as e:
                print(f"[FAKE] parse/handle error: {e}")
                resp = build_resp(0, 0x00, COMM_ERR, b"")

            writer.write(resp)
            await writer.drain()
    except asyncio.IncompleteReadError:
        print(f"[FAKE] client disconnected: {addr}")
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    dev = FakeDioDevice()
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, dev), "127.0.0.1", 2002)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[FAKE] listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
