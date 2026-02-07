# QuantBox Copilot Instructions (Strict Profitable Mode)

이 문서는 바이낸스 선물 페어 트레이딩 시스템 'QuantBox'의 절대적인 개발 지침입니다.
우리의 목표는 **학술적 연구가 아닌, 실제 시장에서 수익을 내는 시스템**을 구축하는 것입니다.

## 🚨 최우선 원칙 (Strict Rules)

1.  **Real Data Only (No Mock Data)**
    - 백테스트와 로직 검증에는 **반드시 실제 바이낸스 과거 데이터**만 사용한다.
    - `np.random` 등을 사용한 가짜(Synthetic) 가격 데이터 생성은 **절대 금지**한다.

2.  **Test-Code Parity (1:1 테스트 강제)**
    - 모든 기능 함수(Function)나 클래스 작성 시, 해당 기능을 검증하는 `pytest` 코드를 **반드시 함께 작성**한다.
    - 테스트 없는 코드는 '미완성'으로 간주한다.
    - `pytest-asyncio`를 사용하여 비동기 함수도 철저히 테스트한다.

3.  **Execution Reality (현실적 체결)**
    - 단순 시장가(Market) 주문보다는, 슬리피지를 최소화하는 지정가(Limit) 추격 로직을 우선 고려한다.
    - 백테스트 수익 계산 시 **거래 수수료(0.05%)** 뿐만 아니라 **펀딩비(Funding Fee)**와 **슬리피지**를 보수적으로 차감해야 한다.

4. **실거래와 백테스트의 일치**
    - 백테스트에서 사용한 로직과 동일한 로직이 실거래에서도 사용되어야 한다. 
    - 실거래와 백테스트 간의 괴리는 절대 발생해서는 안 된다.
    - 백테스트에서만 특별히 유리한 조건을 부여하는 코드는 허용되지 않는다.

코딩 컨벤션 및 타입
Type Hinting: typing.List, typing.Dict 대신 Python 3.9+ 표준인 list[], dict[], tuple[] 사용을 지향한다.
Docstring: Google Style Docstring을 사용하여 입출력 데이터의 형상을 명시한다.
Numba: 루프가 10,000회 이상 반복되는 연산(백테스트)에는 반드시 @numba.jit(nopython=True)를 적용한다.

환경 설정
.env 파일에 BINANCE_API_KEY, BINANCE_API_SECRET이 없으면 실행되지 않도록 pydantic-settings (`quantbox.core.settings`)를 활용하여 검증한다.
로그 레벨은 개발 시 DEBUG, 운영 시 INFO로 설정하되, 주문 체결 로그는 별도 파일로 영구 보존한다.
모든 코드 수정 뒤엔 필요하다고 판단되면 README.md, MANUAL.md의 설치 및 실행 방법을 최신 상태로 유지한다.

**Agent Note:** 사용자는 엄격하며 공격적인 퀀트 트레이더입니다. 코드가 예쁘게 보이는 것보다, **돈을 잃지 않는 안전장치(Safety check)**가 구현되었는지를 가장 중요하게 생각합니다.