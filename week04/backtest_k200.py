"""짝 바꾸기 — 삼성전자 주식선물 × 미니코스피200 선물 (README 6-8)

  python backtest_k200.py            # results_k200/ 에 1분봉 전 설정 + 30분봉 네 방향

왜: 5년 데이터에서 삼성–하이닉스 1분 상관은 2026년(0.63)만 높고 2022~2025년은 0.2 — 페어의 전제가 약했음.
    삼성–코스피200 은 삼성이 지수의 약 20% 라 구조적으로 묶여 있음 (겹치는 253일: 1분 상관 0.64~0.73, 30분 0.94).
    규칙·코드는 그대로, 종목 B 만 바꿈.

가격 맞추기 (코드가 두 다리에 같은 승수 10 을 쓰므로)
  미니코스피200 승수 = 1pt 당 50,000원 = 10 × 5,000  →  B 가격 = K200 지수 × 5,000 으로 넣음
  틱: 미니 0.05pt = 1,250원 → 환산 가격에서 250.  pair_core.tick_size 는 가격대 표라 이 값을 모름 → 여기서만 덮어씀
      (환산 K200 가격은 200만 원 이상, 삼성은 40만 원 이하라 겹치지 않음)
  하이닉스 1계약 자리에 미니 1계약. 삼성 계약 수 = 미니 금액 / (β × 삼성 가격) — 2026 기준 2,500만 ÷ (1.1 × 21만) ≈ 11
수수료: 미니코스피200 도 체결금액의 0.003% 가정 (모름). 호가 비용이 삼성 1틱 0.23% vs 미니 0.01% 라 비용 대부분은 삼성 다리.
데이터: 삼성 = results/pair_1m_full.csv, K200 = week02/results/k200_1m_continuous.csv (연결, 2025-10-10 ~ 2026-10-22)
"""
import sys
from pathlib import Path

import pair_core as pc

HERE = Path(__file__).resolve().parent
_orig_tick = pc.tick_size


def tick_k200_aware(p: float) -> float:
    return 250.0 if p >= 1_000_000 else _orig_tick(p)      # 환산 K200 가격대만 미니 틱


pc.tick_size = tick_k200_aware

import backtest_pair as bp      # noqa: E402  (pc.tick_size 를 덮은 뒤에 불러야 함)
import backtest_30m as b30      # noqa: E402

OUT = HERE / "results_k200"
OUT.mkdir(exist_ok=True)
bp.OUT = OUT
b30.OUT = OUT
DATA = str(HERE / "results" / "pair_k200_1m.csv")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("1m", "both"):
        sys.argv = ["backtest_pair.py", "--data", DATA, "--oos-days", "60"]
        bp.main()
    if which in ("30m", "both"):
        sys.argv = ["backtest_30m.py", "--data", DATA, "--oos-days", "60"]
        b30.main()
