from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional


class DioError(RuntimeError):
    """Base error for DIO client."""


class DioConnectionError(DioError):
    pass


class DioProtocolError(DioError):
    pass


@dataclass
class DioConfig:
    ip: str = "192.168.0.2"
    port: int = 2002
    timeout_s: float = 1.0
    retries: int = 2
    retry_delay_s: float = 0.1

    # Ezi-IO-EN-I8O8 기준: 출력이 8~15번 비트에 매핑됨(매뉴얼 기준)
    output_bit_offset: int = 8


class FASTECH_EthernetDio:
    """
    FASTECH Ezi-IO Ethernet DIO (TCP) 비동기 클라이언트.

    Frame (요청):
      [0]  Header   : 0xAA
      [1]  Length   : Length 이후 바이트 수 = SyncNo(1) + Reserved(1) + FrameType(1) + Data(N)
      [2]  SyncNo
      [3]  Reserved : 0x00
      [4]  FrameType
      [5:] Data

    Frame (응답):
      Header, Length, SyncNo, Reserved, FrameType, (CommStatus 1byte + ReplyData...)
    """

    # Frame types (매뉴얼)
    FT_GET_INPUT = 0xC0
    FT_GET_OUTPUT = 0xC5
    FT_SET_OUTPUT = 0xC6

    def __init__(self, cfg: Optional[DioConfig] = None) -> None:
        self.cfg = cfg or DioConfig()
        self.ip: str = self.cfg.ip
        self.port: int = self.cfg.port

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._sync: int = 0
        self._lock = asyncio.Lock()

    async def connect(self):
        if self._writer is not None:
            return
        try:
            self._reader, self._writer = await asyncio.open_connection(self.ip, self.port)
        except Exception as e:
            raise DioConnectionError(f"TCP connect failed: {e}") from e

    async def disconnect(self):
        if self._writer is None:
            return
        try:
            self._writer.close()
            await self._writer.wait_closed()
        finally:
            self._reader = None
            self._writer = None

    # input 8개 번호(0~7)를 key로, on 여부를 value로 하는 dict를 리턴
    async def get_input(self) -> Dict[int, bool]:
        """
        응답 데이터(매뉴얼): 통신상태 1 + Input 4 + Latch 4 = 9 bytes
        """
        resp = await self._request(self.FT_GET_INPUT, b"")
        comm, data = self._parse_response(resp, expect_ft=self.FT_GET_INPUT)

        if comm != 0x00:
            raise DioProtocolError(f"get_input comm_status=0x{comm:02X}")

        if len(data) < 8:
            raise DioProtocolError(f"get_input reply too short: {len(data)}")

        input_mask32 = int.from_bytes(data[0:4], byteorder="little", signed=False)
        return {i: bool((input_mask32 >> i) & 1) for i in range(8)}

    # output 8개 번호(0~7)를 key로, on 여부를 value로 하는 dict를 리턴
    async def get_output(self) -> Dict[int, bool]:
        """
        과제 제공 테스트 코드가 8~15만 출력하므로 8~15 키로 반환.
        응답 데이터(매뉴얼): 통신상태 1 + Output 4 + Run/Stop 4 = 9 bytes
        """
        resp = await self._request(self.FT_GET_OUTPUT, b"")
        comm, data = self._parse_response(resp, expect_ft=self.FT_GET_OUTPUT)

        if comm != 0x00:
            raise DioProtocolError(f"get_output comm_status=0x{comm:02X}")

        if len(data) < 8:
            raise DioProtocolError(f"get_output reply too short: {len(data)}")

        out_mask32 = int.from_bytes(data[0:4], byteorder="little", signed=False)

        base = self.cfg.output_bit_offset
        return {base + i: bool((out_mask32 >> (base + i)) & 1) for i in range(8)}

    # output 8개 번호(0~7)를 key로, on 여부를 value로 하는 dict를 전달
    async def set_output(self, set_pin_num: List[int] = [], reset_pin_num: List[int] = []):
        """
        송신 데이터(매뉴얼): 8 bytes
          - Set Output (4 bytes mask)
          - Reset Output (4 bytes mask)
        """
        _validate_pins(set_pin_num)
        _validate_pins(reset_pin_num)

        base = self.cfg.output_bit_offset

        set_mask = 0
        for p in set_pin_num:
            set_mask |= 1 << (base + p)

        reset_mask = 0
        for p in reset_pin_num:
            reset_mask |= 1 << (base + p)

        payload = set_mask.to_bytes(4, "little") + reset_mask.to_bytes(4, "little")

        resp = await self._request(self.FT_SET_OUTPUT, payload)
        comm, _data = self._parse_response(resp, expect_ft=self.FT_SET_OUTPUT)

        if comm != 0x00:
            raise DioProtocolError(f"set_output comm_status=0x{comm:02X}")
        return

    # ---------------------------
    # Internal helpers
    # ---------------------------
    async def _request(self, frame_type: int, data: bytes) -> bytes:
        if self._writer is None or self._reader is None:
            raise DioConnectionError("not connected")

        async with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(self.cfg.retries + 1):
                try:
                    self._sync = (self._sync + 1) & 0xFF
                    req = self._build_frame(frame_type, data, sync_no=self._sync)

                    self._writer.write(req)
                    await self._writer.drain()

                    header = await asyncio.wait_for(self._reader.readexactly(2), timeout=self.cfg.timeout_s)
                    if header[0] != 0xAA:
                        raise DioProtocolError(f"bad header: 0x{header[0]:02X}")
                    length = header[1]
                    body = await asyncio.wait_for(self._reader.readexactly(length), timeout=self.cfg.timeout_s)
                    return header + body
                except Exception as e:
                    last_err = e
                    try:
                        await self.disconnect()
                    except Exception:
                        pass
                    try:
                        await self.connect()
                    except Exception:
                        pass
                    if attempt < self.cfg.retries:
                        await asyncio.sleep(self.cfg.retry_delay_s)

            raise DioConnectionError(f"request failed: {last_err}") from last_err

    def _build_frame(self, frame_type: int, data: bytes, sync_no: int) -> bytes:
        reserved = 0x00
        length = 1 + 1 + 1 + len(data)  # sync + reserved + frame_type + data
        if length > 255:
            raise ValueError("frame too long")
        return bytes([0xAA, length, sync_no & 0xFF, reserved, frame_type & 0xFF]) + data

    def _parse_response(self, resp: bytes, expect_ft: int) -> tuple[int, bytes]:
        if len(resp) < 6:
            raise DioProtocolError(f"response too short: {len(resp)} bytes")

        header = resp[0]
        length = resp[1]
        if header != 0xAA:
            raise DioProtocolError(f"bad header: 0x{header:02X}")

        if len(resp) != 2 + length:
            raise DioProtocolError(f"length mismatch: length={length}, got={len(resp)-2}")

        reserved = resp[3]
        frame_type = resp[4]
        if reserved != 0x00:
            raise DioProtocolError(f"reserved not 0x00: 0x{reserved:02X}")
        if frame_type != (expect_ft & 0xFF):
            raise DioProtocolError(f"frame_type mismatch: expect=0x{expect_ft:02X}, got=0x{frame_type:02X}")

        comm_status = resp[5]
        reply_data = resp[6:]
        return comm_status, reply_data


def _validate_pins(pins: List[int]) -> None:
    for p in pins:
        if not isinstance(p, int):
            raise TypeError(f"pin must be int, got {type(p)}")
        if p < 0 or p > 7:
            raise ValueError(f"pin out of range (0~7): {p}")


async def main():
    dio = FASTECH_EthernetDio()
    await dio.connect()

    delay = 0.5
    for pin_num in range(8):
        await dio.set_output(set_pin_num=[pin_num])
        print({n: v for n, v in (await dio.get_output()).items() if 8 <= n <= 15})
        await asyncio.sleep(delay)

    for pin_num in range(8):
        await dio.set_output(reset_pin_num=[pin_num])
        print({n: v for n, v in (await dio.get_output()).items() if 8 <= n <= 15})
        await asyncio.sleep(delay)

    print({n: v for n, v in (await dio.get_input()).items() if 0 <= n <= 8})

    await dio.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
