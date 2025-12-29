# demo_patterns.py

import asyncio
from typing import Dict, List, Tuple

from fastech_dio import FASTECH_EthernetDio, DioConfig


def bits_to_state(bits: str, pin0_is_leftmost: bool = True) -> Dict[int, bool]:
    """
    bits: "11110000" 같은 8자리 문자열

    pin0_is_leftmost=True:
      bits[0] -> pin0, bits[7] -> pin7

    """
    bits = bits.strip()
    if len(bits) != 8 or any(c not in "01" for c in bits):
        raise ValueError("bits must be 8 chars of 0/1")

    if pin0_is_leftmost:
        return {i: (bits[i] == "1") for i in range(8)}
    else:
        return {i: (bits[7 - i] == "1") for i in range(8)}


def diff_only(prev: Dict[int, bool], nxt: Dict[int, bool], pins_to_allow: List[int]) -> Dict[int, bool]:
    """
    '0~1번만 변경' 같은 요구를 확실히 보여주기 위한 헬퍼:
    prev 상태를 기준으로 pins_to_allow만 nxt 값으로 바꾸고 나머지는 prev 유지.
    """
    out = dict(prev)
    for p in pins_to_allow:
        out[p] = nxt[p]
    return out


async def main():
    dio = FASTECH_EthernetDio(DioConfig(ip="127.0.0.1", port=2002))
    await dio.connect()
    print("[DEMO] connect OK")

    inp = await dio.get_input()
    print("[DEMO] get_input:", inp)

    # 1) 11111111
    s1 = bits_to_state("11111111", pin0_is_leftmost=True)
    await dio.set_output(s1)
    out = await dio.get_output()
    print("[DEMO] set_output 11111111 -> get_output:", out)

    # 2) 00000000
    s2 = bits_to_state("00000000", pin0_is_leftmost=True)
    await dio.set_output(s2)
    out = await dio.get_output()
    print("[DEMO] set_output 00000000 -> get_output:", out)

    # 3) 10101010
    s3 = bits_to_state("10101010", pin0_is_leftmost=True)
    await dio.set_output(s3)
    out = await dio.get_output()
    print("[DEMO] set_output 10101010 -> get_output:", out)

    # 4) 01101010 (0~1번만 변경)
    #    “0~1번만 변경”을 진짜로 보여주려면, 직전 상태(out)를 기준(prev)으로
    #    pin0,1만 목표 패턴 값으로 바꾸고 나머지는 prev 유지해서 set_output 호출.
    target = bits_to_state("01101010", pin0_is_leftmost=True)
    prev = out  # 직전 get_output 결과
    s4 = diff_only(prev, target, pins_to_allow=[0, 1])
    await dio.set_output(s4)
    out = await dio.get_output()
    print("[DEMO] set_output 01101010 (change pin0~1 only) -> get_output:", out)

    # 5) 11110000
    s5 = bits_to_state("11110000", pin0_is_leftmost=True)
    await dio.set_output(s5)
    out = await dio.get_output()
    print("[DEMO] set_output 11110000 -> get_output:", out)

    await dio.disconnect()
    print("[DEMO] disconnect OK")


if __name__ == "__main__":
    asyncio.run(main())
