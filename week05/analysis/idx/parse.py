"""funetf ETF 페이지의 구성종목 표(result_pdf)에서 종목·평가금액을 뽑아 비중 csv 로 저장. 실행: python3 parse.py
krx300=KODEX KRX300(292190), k200=KODEX 200(069500), kq150=KODEX 코스닥150(229200)"""
import re, csv
for tag, f in (("krx300", "KR7292190006"), ("k200", "KR7069500007"), ("kq150", "KR7229200001")):
    h = open(f + ".html", encoding="utf-8").read(); body = h[h.find('id="result_pdf"'):]; body = body[:body.find("</tbody>")]
    rows = []
    for tr in body.split("<tr>")[1:]:
        nm = re.search(r'class="ellipsis">(.*?)</p>\s*\((\w+)\)', tr, re.S); td = re.findall(r'<td class="data[^"]*">(.*?)</td>', tr, re.S)
        if not nm or len(td) < 3: continue
        rows.append([nm.group(2), nm.group(1).strip(), float(td[2].replace(",", "") or 0), td[0].strip()])
    tot = sum(r[2] for r in rows); stock = [r for r in rows if re.fullmatch(r"[0-9A-Z]{6}", r[0])]
    csv.writer(open(tag + "_w.csv", "w", newline="", encoding="utf-8")).writerows([["code", "name", "w"]] + [[r[0], r[1], round(r[2] / tot * 100, 4)] for r in stock])
    print(tag, "행", len(rows), "주식", len(stock), "주식 비중 합", round(sum(r[2] for r in stock) / tot * 100, 2), "| 주식 아닌 행:", [(r[0], r[1], round(r[2] / tot * 100, 2)) for r in rows if r not in stock][:5], "| 상위:", [(r[1], round(r[2] / tot * 100, 2)) for r in stock[:3]])
