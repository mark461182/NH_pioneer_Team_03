"""KRX300 과 코스피200 의 구성 차이 (ETF 보유종목 기준, 11/18). 실행: python3 comp.py"""
import pandas as pd
R = {n: pd.read_csv(n + "_w.csv", dtype={"code": str}).set_index("code") for n in ("krx300", "k200", "kq150")}
for n in R: R[n]["w"] = R[n].w / R[n].w.sum() * 100          # 현금 빼고 주식만 100%
x, k, q = R["krx300"], R["k200"], R["kq150"]
T = pd.DataFrame({"name": x.name.combine_first(k.name), "w300": x.w, "w200": k.w}).reindex(x.index.union(k.index)); T["name"] = T.name.fillna(k.name); T = T.fillna({"w300": 0, "w200": 0})
T["act"] = T.w300 - T.w200
T["grp"] = "공통"; T.loc[T.w200 == 0, "grp"] = "KRX300에만(코스닥150)"; T.loc[(T.w200 == 0) & ~T.index.isin(q.index), "grp"] = "KRX300에만(그 외)"; T.loc[T.w300 == 0, "grp"] = "코스피200에만"
T.to_csv("active.csv")
print("종목 수: KRX300", len(x), "코스피200", len(k), "| 공통", int((T.grp == "공통").sum()))
print(T.groupby("grp").agg(종목수=("act", "size"), KRX300비중=("w300", "sum"), 코스피200비중=("w200", "sum"), 차이=("act", "sum")).round(2).to_string())
c = T[T.grp == "공통"]; print("\n공통 종목: KRX300 비중 / 코스피200 비중 의 비율  중앙", round(float((c.w300 / c.w200).median()), 3), "| 범위", round(float((c.w300 / c.w200).quantile(.05)), 3), "~", round(float((c.w300 / c.w200).quantile(.95)), 3))
print("  삼성전자", T.loc["005930", ["w300", "w200", "act"]].astype(float).round(2).to_dict(), "| SK하이닉스", T.loc["000660", ["w300", "w200", "act"]].astype(float).round(2).to_dict())
print("\nKRX300 에만 있는 종목 상위 15:"); print(T[T.w200 == 0].nlargest(15, "w300")[["name", "w300", "grp"]].round(2).to_string())
print("\n코스피200 에만 있는 종목:"); print(T[T.w300 == 0].nlargest(15, "w200")[["name", "w200"]].round(2).to_string())
print("\n비중 차이 큰 순 (KRX300 − 코스피200):"); print(T.reindex(T.act.abs().sort_values(ascending=False).index).head(12)[["name", "w300", "w200", "act"]].round(2).to_string())
print("\n|비중 차이| 합 / 2 (액티브 셰어):", round(float(T.act.abs().sum() / 2), 2), "%")
qq = x.index.intersection(q.index); print("코스닥150 과 겹치는 종목", len(qq), "개, KRX300 안 비중", round(float(x.w[qq].sum()), 2), "%")
