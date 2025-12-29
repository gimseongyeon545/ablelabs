## [구현 방향 및 전략]

### 1) 요구사항 분석

- **목표**: FASTECH Ezi-IO Ethernet DIO 장치에 TCP로 접속하여 입력 8채널 상태 조회, 출력 8채널 상태 조회, 출력 채널 ON/OFF 제어를 Python으로 구현한다.
- **구현 형태**: 클래스 기반, asyncio 비동기, 통신 에러 대응(재시도/재연결) 포함.
- **타겟**: 기본은 Ezi-IO-EN-I8O8을 가정하며, 향후 I32/O32/I16O16로 확장 가능하도록 구성 요소를 분리한다.

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

### 3) 개발(Implementation)

- **통신 프로토콜은 제조사 “통신 기능편”의 프레임 구조를 따른다.**
    - Header(0xAA), Length, SyncNo, Reserved(0x00), FrameType, Data
    - 응답에는 통신상태 1byte가 포함되며, `0x00`이 아닐 경우 예외 처리한다.
- **기능 구현**:
    - `get_input`: 입력 상태 마스크를 파싱하여 채널별 `dict[int,bool]`로 반환
    - `get_output`: 출력 상태 마스크를 파싱하여 0\~7 채널 dict로 반환하고, 필요 시 출력 단계에서 8\~15로 매핑해 표시
    - `set_output`: set/reset 마스크를 생성하여 지정 채널만 ON/OFF 변경

### 4) 테스트(Test)

- **통합 테스트(Integration test):**
    - 제공된 시나리오(출력 순차 ON → 순차 OFF → 입력 조회)를 실행하여 로그로 검증한다.
- **추가 테스트 전략(장비 부재/자동화 대비):**
    - Mock TCP 서버를 사용해 매뉴얼 형식 응답을 주입하고, 프레임 생성/파싱 및 재시도 동작을 검증한다.
    - 단위 테스트: `_build_frame`, `_parse_response`, bit-mask 계산 로직을 pytest로 검증한다.

### 5) 배포(Deploy)

- **실행/배포 형태:**
    - 단일 Python 파일 또는 모듈(`fastech_dio.py`)로 패키징 가능
- **설정 분리:**
    - IP/Port/Timeout/Retry 및 모델 파라미터는 `DioConfig`로 분리하여 운영 환경에서 손쉽게 변경 가능

### 6) 유지보수(Maintain)

- **에러 분류:**
    - Connection 레벨 오류(`DioConnectionError`)와 Protocol/Device 응답 오류(`DioProtocolError`)를 분리하여 장애 원인 파악을 단순화
- **로깅/관측성(추가 권장):**
    - 송신/수신 프레임 덤프(옵션), 재시도 횟수, 통신상태 코드 등을 로그로 남겨 현장 디버깅 시간을 단축
- **확장 계획:**
    - 모델별 입출력 채널 수/offset을 config화하여 I32/O32/I16O16 지원
    - 추가 FrameType(다른 기능)도 동일한 `_request` 공통 경로로 추가 가능
