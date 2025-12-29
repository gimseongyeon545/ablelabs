# [구현 과정 문서]
## 1. 과제 목적
- 본 과제는 FASTECH Ethernet DIO 모듈을 대상으로 TCP 기반 통신 프로토콜을 이해하고, 비동기 방식으로 입력/출력 제어 기능을 구현하는 것을 목표로 한다.
- 실제 장비 제어 환경을 가정하여, 연결 관리, 통신 오류 처리, 테스트 시나리오 실행까지 포함하였다.
- 실제 장비 미보유로 제조사 프로토콜을 모사한 Fake DIO TCP 서버를 구현하여 통신 검증

</br>

## 2. 전체 구조
- 본 구현은 다음 3개의 구성 요소로 이루어진다.
### DIO 제어 클래스 (`fastech_dio.py`)
- DIO 모듈과의 TCP 통신 담당
- 입력/출력 제어 API 제공

### Fake DIO 서버 (`fake_dio_server.py`)
- 실제 DIO 장비를 모사하는 TCP 서버
- 요청 프레임을 파싱하고 내부 상태를 유지하며 응답 반환

### 테스트/데모 코드 (`demo_patterns.py`)
- 과제에서 제시된 출력 패턴 시나리오 실행
- 각 단계별 결과를 로그로 출력

</br>

## 3. 클래스 설계 및 구현
### 3.1 클래스 구현
- FASTECH_EthernetDio 클래스로 구현
- 설정 값은 DioConfig dataclass로 분리하여 가독성과 확장성 확보

### 3.2 비동기 통신
- asyncio.open_connection 기반 비동기 TCP 통신
- async/await 구조 사용
- 동시에 여러 요청이 발생할 수 있는 상황을 고려하여 asyncio.Lock으로 요청 보호

### 3.3 구현된 API
- `connect()` : DIO 모듈과 TCP 연결
- `disconnect()` : 연결 종료
- `get_input()` : 입력 0~7번을 key, on/off 상태를 value로 하는 dict 반환
- `get_output()` : 출력 0~7번을 key, on/off 상태를 value로 하는 dict 반환
- `set_output(state: Dict[int,bool])` : 출력 0~7번에 대한 on/off 상태를 dict 형태로 전달하여 출력 제어

</br>

## 4. 통신 프로토콜 처리
- 제조사 매뉴얼에 정의된 프레임 구조를 그대로 구현
- Header, Length, Sync No, Reserved, Frame Type, Data
- 응답 프레임에 대해 다음을 검증:
  - Header 값
  - Length 일치 여부
  - Sync No 일치 여부
  - Frame Type 일치 여부
- 이상 발생 시 DioProtocolError 예외 발생

</br>

## 5. 통신 오류 처리
- 다음 상황에 대한 예외 처리 구현:
  - TCP 연결 실패
  - 응답 타임아웃
  - 프로토콜 불일치
- 오류 발생 시:
  - 연결 해제
  - 재연결 시도
  - 설정된 횟수만큼 재시도 후 실패 처리

</br>

## 6. 테스트 방법 및 시나리오
### 6.1 Fake DIO 서버
- 실제 장비가 없는 환경을 고려하여 TCP 기반 Fake DIO 서버 구현
- 내부적으로 출력 상태를 유지하며, set_output 요청에 따라 상태 변경
- `get_output` 요청 시 현재 상태를 응답으로 반환

### 6.2 테스트 시나리오
- 과제에서 제시된 테스트 순서를 그대로 실행하였다.
1. connect
2. get_input
3. set_output(11111111) → get_output
4. set_output(00000000) → get_output
5. set_output(10101010) → get_output
6. set_output(01101010, 0~1번만 변경) → get_output
7. set_output(11110000) → get_output
8. disconnect

</br>

## 7. 작동 로그
- Fake DIO 서버 실행 로그
- 테스트 코드 실행 로그를 통해:
  - TCP 연결 성공 여부
  - 출력 패턴 변경 결과
  - 요청/응답 흐름을 확인함
- 터미널1 로그
  > <img width="912" height="172" alt="Image" src="https://github.com/user-attachments/assets/ff51cbdb-a074-4889-8713-2adacd99a214" />
- 터미널2 로그
  > <img width="1682" height="270" alt="Image" src="https://github.com/user-attachments/assets/78cd1da4-f4e5-49e4-894e-02fa2cbf4a09" />

</br>

## 8. 결론
- 본 과제를 통해 DIO 모듈의 통신 프로토콜을 기반으로 비동기 TCP 제어 클래스 설계, 오류 처리, 테스트 시나리오 검증까지 수행하였다.
- 실제 장비 환경에서도 동일한 구조로 적용 가능하도록 구현하였다.
