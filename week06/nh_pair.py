"""NH 앱(nh-futures-open-api) 과 붙는 부분 — 종목 찾기, 1분봉 받기, 모의 주문 브로커.

시세는 실전(real) 키, 주문은 모의(paper) 키. 팀 모의 키에 [국내 시세조회] 권한이 없어서입니다
(2주차 README 8-1절, HTTP 403). 주문 함수는 environment="paper" 로만 부릅니다.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

QUOTE_ENV = "real"     # --quote-env paper 로 바꿀 수 있음 (모의 키에 [국내 시세조회] 권한이 있을 때)
ORDER_ENV = "paper"    # 바꾸지 말 것


def set_quote_env(env: str) -> None:
    global QUOTE_ENV
    assert env in ("real", "paper")
    QUOTE_ENV = env
KST = dt.timezone(dt.timedelta(hours=9))

# 종목 마스터의 한글종목명 앞부분 (예: "삼성전자   F 202612 (  10)")
NAME_A, NAME_B = "삼성전자", "SK하이닉스"


def load_app(app_dir: str):
    if not app_dir:
        sys.exit("--app-dir 에 nh-futures-open-api 폴더 경로를 주세요")
    sys.path.insert(0, str(Path(app_dir).resolve()))
    from backend.domestic_futureoption import domestic_futureoption_functions as F
    return F


# ── 종목 찾기 ────────────────────────────────────────────────
def master_rows(app_dir: str) -> list[dict]:
    """국내 개별주식 선물 마스터(stk_fut_opt) 의 선물 줄. 오늘 받은 게 없으면 내려받음."""
    load_app(app_dir)
    from backend.master.master_download import download_all
    from backend.master.master_store import load_all
    try:
        download_all()                     # 오늘 이미 있으면 건너뜀
    except Exception as e:                 # 못 받아도 어제 받아 둔 것으로
        print("마스터 내려받기 실패 — 받아 둔 파일 사용:", str(e)[:150])
    rows = []
    for lm in load_all():
        if lm.table is None or lm.table.name != "stk_fut_opt":
            continue
        for raw in lm.table.rows:
            r = dict(zip(lm.table.fields, raw))
            if r.get("fo_tp") == "F" and r.get("opt_tp") == "1" and r.get("cme_tp", "day") != "night":
                rows.append(r)
    return rows


def pick_contract(rows: list[dict], name: str, today: dt.date, min_days: int = 5) -> dict:
    """이름이 `name` 으로 시작하는 선물 중, 만기까지 `min_days` 일 넘게 남은 가장 가까운 월물.

    「삼성전자」 는 「삼성전자우」 와 앞부분이 같으므로 이름 바로 뒤가 공백인 것만.
    만기 주(週)에는 거래가 다음 월물로 옮겨 가므로 5일 미만 남은 근월물은 건너뜀.
    """
    mine = [r for r in rows if r["name_kor"].startswith(name + " ")]
    if not mine:
        raise SystemExit(f"마스터에서 「{name}」 선물을 못 찾음 — --sym-a/--sym-b 로 직접 주세요")
    live = []
    for r in mine:
        exp = dt.datetime.strptime(r["exp_ym"][:8], "%Y%m%d").date()   # 이 파일은 만기일 8자리
        if (exp - today).days >= min_days:
            live.append((exp, r))
    if not live:
        raise SystemExit(f"「{name}」 선물 중 만기가 {min_days}일 넘게 남은 월물이 마스터에 없음 — --sym-a/--sym-b 로 직접 주세요")
    exp, r = min(live, key=lambda x: x[0])
    return {"sym": r["series"].strip(), "name": " ".join(r["name_kor"].split()), "expiry": str(exp),
            "mult": int(float(r.get("ctrt_size") or 10))}


# ── 1분봉 ────────────────────────────────────────────────────
def day_bars(F, sym: str, day: str, continuous: bool = False, retries: int = 3) -> pd.DataFrame:
    """토큰 만료(EGW023 등)·일시 오류면 토큰을 버리고 다시 받아 최대 `retries` 번 재시도."""
    last = None
    for k in range(retries):
        try:
            return _day_bars(F, sym, day, continuous)
        except Exception as e:                        # 앱이 한 번 재발급해도 안 될 때 — 긴 수집에서 한 시간마다 만료됨
            last = e
            msg = str(e)
            print(f"  [{day} {sym}] 재시도 {k + 1}/{retries}: {msg[:120]}")
            try:
                from backend.nh_api.runtime import client as default_client
                default_client()._session.tokens.forget("domestic", QUOTE_ENV)
            except Exception:
                pass
            time.sleep(3 * (k + 1))
    raise last


def _day_bars(F, sym: str, day: str, continuous: bool = False) -> pd.DataFrame:
    """하루 주간 1분봉. index = 봉 시각(KST, tz 없음), columns open/high/low/close/volume.

    continuous=True 면 연결선물(cidx_yn=1) — 만기 지난 월물이 섞인 과거 백테스트용 (2주차 fetch_minute.py 와 같음).
    """
    if continuous:
        from backend.nh_api.runtime import client as default_client   # 앱 함수는 cidx_yn=0 고정이라 직접 부름
        body = default_client().call("domestic_chart-minute", {
            "sym": sym, "inq_strt_dt": day, "inq_ed_dt": day, "strt_tm": "000000", "ed_tm": "235959",
            "dcnt": "1", "nght_tp": "1", "cidx_yn": "1", "fut_cidx_exp_mod": "0",
            "nxt_key": "99999999", "req_qty": "2000"}, environment=QUOTE_ENV).body
    else:
        body = F.chart_minute(sym, start_date=day, end_date=day, rows="2000", environment=QUOTE_ENV)
    rows = [r for r in body.get("Occurs1") or [] if r.get("occurs1_ref_dt") == day]
    return to_frame(rows)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    d = pd.DataFrame(rows)
    idx = pd.to_datetime(d["occurs1_ref_dt"] + d["occurs1_exch_tm"].str.zfill(6), format="%Y%m%d%H%M%S")
    f = pd.DataFrame({k: pd.to_numeric(d[f"occurs1_{k}_pric"], errors="coerce") for k in ("open", "high", "low", "close")})
    f["volume"] = pd.to_numeric(d.get("occurs1_cum_trd_qty"), errors="coerce")
    f.index = idx
    f = f[~f.index.duplicated()].sort_index()
    t = f.index.strftime("%H%M")
    return f[(t >= "0900") & (t < "1535")]          # 15:35~15:45 종가 단일가는 뺌


def quote(F, sym: str) -> dict:
    """현재가·매수호가·매도호가 (실전 시세)."""
    r = F.price(sym, environment=QUOTE_ENV)
    num = lambda k: float(str(r.get(k) or "nan").strip() or "nan")
    return {"last": num("last_pric"), "bid": num("bid_pric"), "ask": num("ask_pric"),
            "mult": num("mltplr"), "name": str(r.get("sym_nm", "")).strip()}


# ── 모의 주문 브로커 (pair_core.PairExecutor 가 부름) ─────────
SIDE = {"buy": "1", "sell": "2"}


def _n(x) -> float:
    try:
        return float(str(x).strip() or "nan")
    except ValueError:
        return float("nan")


class NHPaperBroker:
    """시장가 주문·체결 조회·취소·잔고. 실패는 예외로 올려 PairExecutor 가 기록하게 함."""

    def __init__(self, F, log=None):
        self.F, self.log = F, log or (lambda *a, **k: None)
        self._hist, self._hist_t = None, 0.0

    def send(self, sym: str, side: str, qty: int, price: float | None = None) -> str:
        """price 없음 = 시장가(주문가격 0, 명세). price 있음 = 지정가(ord_tp_cd=1)."""
        assert ORDER_ENV == "paper"
        if price is None:
            r = self.F.order(sym, trd_div=SIDE[side], ord_tp_cd="2", ord_qty=str(qty), ord_pric="0",
                             environment=ORDER_ENV)
        else:
            r = self.F.order(sym, trd_div=SIDE[side], ord_tp_cd="1", ord_qty=str(qty),
                             ord_pric=f"{price:.2f}", environment=ORDER_ENV)   # 국내 가격은 소수 둘째 자리까지 실림
        no = str(r.get("ord_no", "")).strip()
        msg = (str(r.get("msg_cd", "")).strip() + " " + str(r.get("msg_cntn", "")).strip()).strip()
        if not no.strip("0") or msg:                          # 성공이면 메시지 두 칸이 빈 값 (앱 저장 응답)
            raise RuntimeError(f"주문 거부 {sym} {side} {qty}: {msg or '주문번호 없음'}")
        return no

    def _history(self) -> list[dict]:
        if self._hist is None or time.time() - self._hist_t > 0.8:   # 같은 순간 두 다리 조회는 한 번만
            self._hist = self.F.order_history(environment=ORDER_ENV).get("Occurs1") or []
            self._hist_t = time.time()
        return self._hist

    def filled(self, no: str) -> tuple[int, float]:
        for x in self._history():
            if str(x.get("occurs1_ord_no", "")).strip() == no:
                q = _n(x.get("occurs1_exec_qty"))
                px = _n(x.get("occurs1_avg_exec_pric"))
                if px != px or px <= 0:
                    px = _n(x.get("occurs1_exec_pric"))
                return (int(q) if q == q else 0), (px if px == px else 0.0)
        return 0, 0.0

    def cancel(self, no: str, sym: str, side: str) -> None:
        q, _ = self.filled(no)
        for x in self._history():
            if str(x.get("occurs1_ord_no", "")).strip() == no:
                rem = _n(x.get("occurs1_rem_ord_qty"))
                if rem == rem and rem > 0:
                    self.F.cancel(sym, org_ord_no=no, trd_div=SIDE[side], ord_qty=str(int(rem)),
                                  environment=ORDER_ENV)
                return

    def max_qty(self, sym: str, side: str, px: float) -> int:
        """주문가능수량 (모의계좌 증거금 기준). 리스크 레이어 ④."""
        r = self.F.orderable_quantity(sym, trd_div=SIDE[side], ord_pric=f"{px:.2f}", environment=ORDER_ENV)
        q = _n(r.get("tot_psb_qty"))
        return int(q) if q == q else 0

    def net_positions(self) -> dict[str, int]:
        """종목 → 순포지션(매수 +). 주간(KRX) 줄만."""
        r = self.F.open_interest(environment=ORDER_ENV)
        out: dict[str, int] = {}
        for x in r.get("Occurs1") or []:
            if str(x.get("occurs1_exch_cd", "KRX")).strip() not in ("", "KRX"):
                continue
            sym = str(x.get("occurs1_sym", "")).strip()
            q = _n(x.get("occurs1_opint_qty"))
            q = int(q) if q == q else 0
            out[sym] = out.get(sym, 0) + (q if str(x.get("occurs1_trd_div", "")).strip() == "1" else -q)
        return out
