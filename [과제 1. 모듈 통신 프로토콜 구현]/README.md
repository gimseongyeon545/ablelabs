## [구현 방향 및 전략]

### 1) 요구사항 분석

- **목표**: FASTECH Ezi-IO Ethernet DIO 장치에 TCP로 접속하여 입력 8채널 상태 조회, 출력 8채널 상태 조회, 출력 채널 ON/OFF 제어를 Python으로 구현한다.
- **구현 형태**: 클래스 기반, asyncio 비동기, 통신 에러 대응(재시도/재연결) 포함.
- **타겟**: 기본은 Ezi-IO-EN-I8O8을 가정하며, 향후 I32/O32/I16O16로 확장 가능하도록 구성 요소를 분리한다.

</br>

### 2) 설계(Architecture)

- **단일 책임 분리**:
    - `connect/disconnect`: 연결 라이프사이클 관리
    - `get_input/get_output/set_output`: 기능 API
    - `_build_frame/_parse_response`: 프로토콜 프레임 생성/파싱
    - `_request`: 통신 공통 로직(락/타임아웃/재시도)
- **동시성 설계**:
    - 장비는 “요청-응답” 순서가 중요하므로, 내부적으로 `asyncio.Lock`을 사용해 동시 호출 시 프레임이 섞이지 않도록 직렬화한다.
    
- **확장 설계**:
    - 모델별 차이(출력 비트 offset 등)는 `DioConfig`에서 파라미터화한다.
    - 추후 `num_inputs/num_outputs/input_offset/output_offset` 등을 추가해 I32/O32로 확장 가능하게 한다.

</br>

### 3) 개발(Implementation)

- **통신 프로토콜은 제조사 “통신 기능편”의 프레임 구조를 따른다.**
    - Header(0xAA), Length, SyncNo, Reserved(0x00), FrameType, Data
    - 응답에는 통신상태 1byte가 포함되며, `0x00`이 아닐 경우 예외 처리한다.
    - 응답 SyncNo가 요청 SyncNo와 일치하는지 검증한다.
- **기능 구현**:
    - `get_input`: 입력 상태 마스크를 파싱하여 채널별 `dict[int,bool]`로 반환
    - `get_output`: 출력 상태 마스크를 파싱하여 0\~7 채널 dict로 반환하고, 필요 시 출력 단계에서 8\~15로 매핑해 표시
    - `set_output`: set/reset 마스크를 생성하여 지정 채널만 ON/OFF 변경

</br>

### 4) 테스트(Test)

- **수동 시나리오 테스트(장비 연동 환경):**
  - main() 예제 시나리오(출력 순차 ON → 순차 OFF → 입력 조회)를 실행하여 동작을 확인한다. (실장비/네트워크 환경 필요)

- **단위 테스트(pytest, 장비 없이 가능):**
  - `test_fastech_dio.py`에서 _build_frame, _parse_response(Header/Length/FrameType/SyncNo 검증), 출력 bit-mask 매핑 로직을 pytest로 검증한다.
  - 실행 결과
    > <img width="1056" height="88" alt="Image" src="https://github.com/user-attachments/assets/ef3b7e06-8e42-4bc1-960f-401517d3b49e" />

</br>

### 5) 배포(Deploy)

- **실행/배포 형태:**
    - 클라이언트 모듈(ablelabs_robot_control_test.py) + 테스트 파일(test_fastech_dio.py) 형태로 구성한다.
- **설정 분리:**
    - IP/Port/Timeout/Retry 및 모델 파라미터는 DioConfig로 분리하여 운영 환경에서 손쉽게 변경 가능하다.

</br>

### 6) 유지보수(Maintain)

- **에러 분류:**
  - Connection 레벨 오류(DioConnectionError)와 Protocol/Device 응답 오류(DioProtocolError)를 분리하여 장애 원인 파악을 단순화한다.

- **프로토콜 안전장치:**
  - 응답 Length 일치 여부 및 SyncNo 일치 여부를 검증하여 프레임 혼선/손상 시 즉시 예외 처리한다.

- **관측성(현재 구현 범위):**
  - 현재 구현에는 별도 로깅/프레임 덤프 기능은 포함하지 않았다. (필요 시 송수신 프레임 덤프/재시도 카운트 로깅 옵션을 추가 가능)

- **확장 계획(현재 구현 + 향후):**
  - I8O8의 출력 비트 오프셋은 DioConfig.output_bit_offset으로 파라미터화했다.
  - 향후 모델별 채널 수/offset을 DioConfig에 확장하고, 기존 _request 공통 경로에 FrameType을 추가하는 방식으로 기능을 확장 가능하다.
