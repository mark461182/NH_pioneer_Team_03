"""KRX300 선물 시장 분석용 내려받기 (조회만, 주문 함수 안 부름). 앱 로컬 API 사용.
실행(VM): python3 krx_dl.py   → ~/nh_week05/krx300/data/
"""
import json,urllib.request,datetime as D,time,os,collections
B='http://127.0.0.1:47100/api/domestic'; K=D.timezone(D.timedelta(hours=9))
OUT=os.path.expanduser('~/nh_week05/krx300/data'); os.makedirs(OUT,exist_ok=True)
def post(p,body):
    r=urllib.request.Request(B+p,data=json.dumps(body).encode(),headers={'content-type':'application/json'})
    for i in range(3):
        try:
            with urllib.request.urlopen(r,timeout=30) as f: return f.status,json.loads(f.read())
        except urllib.error.HTTPError as e: res=(e.code,e.read().decode()[:200])
        except Exception as e: res=(0,str(e)[:200])
        time.sleep(2)
    return res
SY={'krx300':'A086C000','krx300_mar':'A0873000','k200':'A016C000','mini':'A056C000','kq150':'A066C000'}
# 0) 상품 정보·현재가·호가·투자자
s,c=post('/product-detail',{'sym':'A086C000'}); json.dump(c,open(OUT+'/product_detail.json','w'),ensure_ascii=False)
if s==200:
    for p in c.get('products',[]):
        if p.get('kind')=='futures' and p.get('session')=='day': print('상품',p.get('name'),p.get('productCode'),'tick',p.get('tick'),'tickValue',p.get('tickValue'),'mult',p.get('multiplier'),'증거금',p.get('initialMarginRate'),p.get('contractInitialMargin'),flush=True)
snap={}
for n,sym in SY.items():
    snap[n]={'price':post('/price',{'sym':sym})[1],'book':post('/price-bid',{'sym':sym})[1]}; time.sleep(0.3)
for prd in ('08','01','05','06'):
    snap['inv_'+prd]=post('/trend-investor',{'prd':prd}); time.sleep(0.3)
json.dump(snap,open(OUT+'/snapshot_%s.json'%D.datetime.now(K).strftime('%Y%m%d_%H%M'),'w'),ensure_ascii=False)
print('투자자 08:',json.dumps(snap['inv_08'],ensure_ascii=False)[:700],flush=True)
# 1) 일봉
for n,sym in SY.items():
    rows={}; key=None
    for _ in range(10):
        body={'sym':sym,'inq_strt_dt':'20250101','inq_ed_dt':'20261119'}
        if key: body['nxt_key']=key
        s,c=post('/chart-daily',body)
        if s!=200: print('일봉 실패',n,s,c); break
        b=c.get('bars') or []
        new=[x for x in b if x['timestamp'] not in rows]
        for x in b: rows[x['timestamp']]=x
        key=str(c.get('nextKey') or '').strip(); time.sleep(0.3)
        if not new or not key: break
    R=[rows[k] for k in sorted(rows)]; json.dump(R,open(f'{OUT}/{n}_1d.json','w'))
    d=lambda x:D.datetime.fromtimestamp(x['timestamp']/1000,K).date()
    print('일봉',n,len(R),d(R[0]) if R else '',d(R[-1]) if R else '','거래있는날',sum(1 for x in R if x['volume']>0),flush=True)
# 2) 1분봉 (하루씩, 페이지 끝까지)
def minute(sym,ds):
    rows={}; key=None
    for _ in range(8):
        body={'sym':sym,'inq_strt_dt':ds,'inq_ed_dt':ds,'dcnt':'1'}
        if key: body['nxt_key']=key
        s,c=post('/chart-minute',body)
        if s!=200: return rows,(s,str(c)[:80])
        b=c.get('bars') or []
        new=[x for x in b if x['timestamp'] not in rows]
        for x in b: rows[x['timestamp']]=x
        key=str(c.get('nextKey') or '').strip(); time.sleep(0.25)
        if not new or not key or key[:8]<ds: break
    return rows,None
for n in ('krx300','k200','mini','krx300_mar'):
    sym=SY[n]; allb={}; bad=0; d=D.date(2026,11,19); errs=[]
    while d>=D.date(2026,6,1) and bad<7:
        if d.weekday()<5:
            r,e=minute(sym,d.strftime('%Y%m%d'))
            if e: errs.append((str(d),e))
            allb.update(r); bad = bad+1 if not r else 0
        d-=D.timedelta(days=1)
    R=[allb[k] for k in sorted(allb)]; json.dump(R,open(f'{OUT}/{n}_1m.json','w'))
    day=lambda b:D.datetime.fromtimestamp(b['timestamp']/1000,K).date()
    cnt=collections.Counter(map(day,R)); days=sorted(cnt)
    print('분봉',n,len(R),'일수',len(days),days[0] if days else '','~',days[-1] if days else '','오류',errs[:4],flush=True)
print('DONE',flush=True)
