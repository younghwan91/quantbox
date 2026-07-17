"""리서치 지원 코드 — 실험 스크립트가 import하는 진짜 패키지.

`research/active/**` 의 실험 스크립트는 sys.path 해킹 없이 여기서 import한다.
라이브 런타임은 이 패키지에 의존하지 않는다 (의존 방향: research -> quantbox, 역방향 금지).
"""
