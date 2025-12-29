# test_fastech_dio.py
import pytest
from ablelabs_robot_control_test import FASTECH_EthernetDio, DioConfig, DioProtocolError


def _make_client() -> FASTECH_EthernetDio:
    # 테스트는 네트워크 연결 안 하므로 ip/port는 의미 없음
    return FASTECH_EthernetDio(DioConfig(ip="127.0.0.1", port=2002))


def test_build_frame_basic():
    dio = _make_client()

    frame_type = dio.FT_GET_INPUT  # 0xC0
    data = b""
    sync = 0x10

    fr = dio._build_frame(frame_type, data, sync_no=sync)

    # [0]=0xAA, [1]=Length, [2]=Sync, [3]=0x00, [4]=FrameType
    assert fr[0] == 0xAA
    assert fr[1] == 3  # Sync(1)+Reserved(1)+FrameType(1)+Data(0)
    assert fr[2] == sync
    assert fr[3] == 0x00
    assert fr[4] == (frame_type & 0xFF)
    assert fr[5:] == b""


def test_parse_response_ok_and_extract_comm_and_data():
    dio = _make_client()

    expect_ft = dio.FT_GET_OUTPUT  # 0xC5
    expect_sync = 0x22

    comm_status = 0x00
    reply_data = b"\x01\x02\x03\x04\xAA\xBB\xCC\xDD"  # 임의 데이터

    # 응답 body 구성: SyncNo + Reserved + FrameType + CommStatus + ReplyData
    body = bytes([expect_sync, 0x00, expect_ft, comm_status]) + reply_data
    length = len(body)  # Length 이후 바이트 수
    resp = bytes([0xAA, length]) + body

    comm, data = dio._parse_response(resp, expect_ft=expect_ft, expect_sync=expect_sync)
    assert comm == 0x00
    assert data == reply_data


def test_parse_response_rejects_bad_header():
    dio = _make_client()

    expect_ft = dio.FT_GET_INPUT
    expect_sync = 0x01

    body = bytes([expect_sync, 0x00, expect_ft, 0x00])  # CommStatus=0
    resp = bytes([0xAB, len(body)]) + body

    with pytest.raises(DioProtocolError):
        dio._parse_response(resp, expect_ft=expect_ft, expect_sync=expect_sync)


def test_parse_response_rejects_length_mismatch():
    dio = _make_client()

    expect_ft = dio.FT_GET_INPUT
    expect_sync = 0x01

    body = bytes([expect_sync, 0x00, expect_ft, 0x00])
    resp = bytes([0xAA, len(body) + 1]) + body

    with pytest.raises(DioProtocolError):
        dio._parse_response(resp, expect_ft=expect_ft, expect_sync=expect_sync)


def test_parse_response_rejects_sync_mismatch():
    dio = _make_client()

    expect_ft = dio.FT_GET_INPUT
    expect_sync = 0x10
    wrong_sync = 0x11

    body = bytes([wrong_sync, 0x00, expect_ft, 0x00])
    resp = bytes([0xAA, len(body)]) + body

    with pytest.raises(DioProtocolError):
        dio._parse_response(resp, expect_ft=expect_ft, expect_sync=expect_sync)


def test_parse_response_rejects_frame_type_mismatch():
    dio = _make_client()

    expect_ft = dio.FT_GET_INPUT
    expect_sync = 0x10
    wrong_ft = dio.FT_GET_OUTPUT

    body = bytes([expect_sync, 0x00, wrong_ft, 0x00])
    resp = bytes([0xAA, len(body)]) + body

    with pytest.raises(DioProtocolError):
        dio._parse_response(resp, expect_ft=expect_ft, expect_sync=expect_sync)


def test_set_output_mask_mapping_i8o8_offset():
    """
    장비 없이도 '마스크 계산이 맞는지'는 검증 가능.
    I8O8은 Output이 8~15 비트에 매핑되므로,
    set_pin_num=[0]이면 set_mask의 bit8이 1이어야 한다.
    """
    dio = _make_client()
    base = dio.cfg.output_bit_offset  # 8

    set_pin_num = [0, 3, 7]
    set_mask = 0
    for p in set_pin_num:
        set_mask |= 1 << (base + p)

    assert (set_mask >> 8) & 1 == 1   # pin0 -> bit8
    assert (set_mask >> 11) & 1 == 1  # pin3 -> bit11
    assert (set_mask >> 15) & 1 == 1  # pin7 -> bit15
