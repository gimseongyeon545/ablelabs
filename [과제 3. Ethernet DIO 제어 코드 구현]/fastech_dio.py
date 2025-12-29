# fastech_dio.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional


class DioError(RuntimeError):
    """Base error for DIO client."""


class DioConnectionError(DioError):
    pass


class DioProtocolError(DioError):
    pass


@dataclass
class DioConfig:
    ip: str = "127.0.0.1"
    port: int = 2002
    timeout_s: float = 1.0
    retries: int = 2
    retry_delay_s: float = 0.1

    # Ezi-IO-EN-I8O8 기준: 출력이 8~15번 비트에 매핑됨
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

    async def connect(self) -> None:
        if self._writer is not None:
            return
        try:
            self._reader, self._writer = await asyncio.open_connection(self.ip, self.port)
        except Exception as e:
            raise DioConnectionError(f"TCP connect failed: {e}") from e

    async def disconnect(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
            await self._writer.wait_closed()
        finally:
            self._reader = None
            self._writer = None

    async def get_input(self) -> Dict[int, bool]:
        """
        응답 데이터(매뉴얼): 통신상태 1 + Input 4 + Latch 4 = 9 bytes
        여기서는 reply_data 최소 8바이트(4+4)만 있으면 파싱 가능하게 처리.
        """
        comm, data = await self._request(self.FT_GET_INPUT, b"")
        if comm != 0x00:
            raise DioProtocolError(f"get_input comm_status=0x{comm:02X}")
        if len(data) < 8:
            raise DioProtocolError(f"get_input reply too short: {len(data)}")

        input_mask32 = int.from_bytes(data[0:4], byteorder="little", signed=False)
        return {i: bool((input_mask32 >> i) & 1) for i in range(8)}

    async def get_output(self) -> Dict[int, bool]:
        """
        외부 API는 0~7 출력 핀 기준으로 반환.
        (I8O8 내부 매핑은 output_bit_offset(기본 8)을 이용해 8~15 비트를 읽음)
        응답 데이터(매뉴얼): 통신상태 1 + Output 4 + Run/Stop 4 = 9 bytes
        """
        comm, data = await self._request(self.FT_GET_OUTPUT, b"")
        if comm != 0x00:
            raise DioProtocolError(f"get_output comm_status=0x{comm:02X}")
        if len(data) < 8:
            raise DioProtocolError(f"get_output reply too short: {len(data)}")

        out_mask32 = int.from_bytes(data[0:4], byteorder="little", signed=False)
        base = self.cfg.output_bit_offset
        return {i: bool((out_mask32 >> (base + i)) & 1) for i in range(8)}

    async def set_output(self, state: Dict[int, bool]) -> None:
        """
        과제 요구사항 그대로:
          output 0~7을 key, on/off를 value로 하는 dict를 전달.
        송신 데이터(매뉴얼): 8 bytes
          - Set Output (4 bytes mask)
          - Reset Output (4 bytes mask)
        """
        self._validate_state(state)

        base = self.cfg.output_bit_offset
        set_mask = 0
        reset_mask = 0

        # dict에 없는 핀은 "변경하지 않음"으로 두고 싶으면 정책을 바꿔야 함.
        # 과제 패턴(11110000 등)은 보통 전체를 세팅하므로, 여기서는 "0~7 모두 들어온다" 전제로 함.
        # (demo에서 항상 0~7 전체를 넣어준다.)
        for pin in range(8):
            v = bool(state[pin])
            if v:
                set_mask |= 1 << (base + pin)
            else:
                reset_mask |= 1 << (base + pin)

        payload = set_mask.to_bytes(4, "little") + reset_mask.to_bytes(4, "little")
        comm, _ = await self._request(self.FT_SET_OUTPUT, payload)
        if comm != 0x00:
            raise DioProtocolError(f"set_output comm_status=0x{comm:02X}")

    async def _request(self, frame_type: int, data: bytes) -> tuple[int, bytes]:
        if self._writer is None or self._reader is None:
            raise DioConnectionError("not connected")

        async with self._lock:
            last_err: Optional[Exception] = None

            for attempt in range(self.cfg.retries + 1):
                try:
                    self._sync = (self._sync + 1) & 0xFF
                    sync = self._sync

                    req = self._build_frame(frame_type, data, sync_no=sync)
                    self._writer.write(req)
                    await self._writer.drain()

                    header = await asyncio.wait_for(
                        self._reader.readexactly(2), timeout=self.cfg.timeout_s
                    )
                    if header[0] != 0xAA:
                        raise DioProtocolError(f"bad header: 0x{header[0]:02X}")

                    length = header[1]
                    body = await asyncio.wait_for(
                        self._reader.readexactly(length), timeout=self.cfg.timeout_s
                    )
                    resp = header + body
                    return self._parse_response(resp, expect_ft=frame_type, expect_sync=sync)

                except (asyncio.TimeoutError, OSError, ConnectionError) as e:
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

                except DioProtocolError:
                    raise

                except Exception as e:
                    last_err = e
                    raise DioConnectionError(f"request failed: {e}") from e

            raise DioConnectionError(f"request failed: {last_err}") from last_err

    def _build_frame(self, frame_type: int, data: bytes, sync_no: int) -> bytes:
        reserved = 0x00
        length = 1 + 1 + 1 + len(data)  # SyncNo + Reserved + FrameType + Data
        if length > 255:
            raise ValueError("frame too long")
        return bytes([0xAA, length, sync_no & 0xFF, reserved, frame_type & 0xFF]) + data

    def _parse_response(
        self, resp: bytes, expect_ft: int, expect_sync: Optional[int] = None
    ) -> tuple[int, bytes]:
        if len(resp) < 6:
            raise DioProtocolError(f"response too short: {len(resp)} bytes")

        header = resp[0]
        length = resp[1]
        if header != 0xAA:
            raise DioProtocolError(f"bad header: 0x{header:02X}")

        if len(resp) != 2 + length:
            raise DioProtocolError(f"length mismatch: length={length}, got={len(resp)-2}")

        sync_no = resp[2]
        if expect_sync is not None and sync_no != (expect_sync & 0xFF):
            raise DioProtocolError(
                f"sync mismatch: expect=0x{expect_sync:02X}, got=0x{sync_no:02X}"
            )

        reserved = resp[3]
        frame_type = resp[4]
        if reserved != 0x00:
            raise DioProtocolError(f"reserved not 0x00: 0x{reserved:02X}")
        if frame_type != (expect_ft & 0xFF):
            raise DioProtocolError(
                f"frame_type mismatch: expect=0x{expect_ft:02X}, got=0x{frame_type:02X}"
            )

        comm_status = resp[5]
        reply_data = resp[6:]
        return comm_status, reply_data

    @staticmethod
    def _validate_state(state: Dict[int, bool]) -> None:
        if not isinstance(state, dict):
            raise TypeError("state must be dict[int,bool]")
        for k, v in state.items():
            if not isinstance(k, int):
                raise TypeError(f"pin key must be int, got {type(k)}")
            if k < 0 or k > 7:
                raise ValueError(f"pin out of range (0~7): {k}")
            if not isinstance(v, bool):
                raise TypeError(f"pin value must be bool, got {type(v)}")
        # 과제 패턴 세팅용: 0~7 전체가 있어야 “절대값” 세팅 가능
        for pin in range(8):
            if pin not in state:
                raise ValueError(f"state must include pin {pin} (0~7 all required)")
