# 1주차 · 팀 GitHub 준비

**제출일: 2026/10/23(금) 23:59 까지**

팀 저장소를 만들고 팀 Discord 방에 저장소 링크를 올립니다.

## 제출 체크리스트

- [x] 계좌로 APP Key·Secret Key 발급
- [x] 발급받은 키로 API 한 번 호출해 응답 확인
- [x] 팀 GitHub 저장소 생성
- [x] README에 팀명과 간단한 소개 작성
- [x] 운영진의 저장소 접근 가능 여부 확인 (Public)
- [ ] 10/23(금) 23:59 까지 팀 Discord 방에 저장소 링크 올리기

## Discord에 올릴 문구

```text
[Pioneer 1기 · 1주차]
팀명: 풀옵션
GitHub 저장소: https://github.com/mark461182/NH_pioneer_Team_03
```

## 이번 주차에 올린 것

프로그램·보고서 제출은 1주차에 포함하지 않지만, 첫 호출에 쓴 코드와 응답을 기록으로 남겼습니다.

| 파일 | 내용 |
| --- | --- |
| `first_call.py` | 접근토큰 발급 → 국내 현재가 조회 → 응답 저장 |
| `config.example.yaml` | 설정 파일 서식 (키는 비어 있음) |
| `results/` | 실행 응답 원본 (JSON) |

### 호출한 API

| 용도 | 경로 | 방식 |
| --- | --- | --- |
| 접근토큰 발급 | `POST /auth-service/v1/token` | form — `grant_type=client_credentials`, `appkey`, `appsecret` |
| 국내 현재가 | `POST /trade/v1/domestic/price` | JSON — `{"sym": "A016C000"}`, 헤더 `Authorization: Bearer {token}` |

모의환경(`https://apidemo.futures.co.kr`)에서 코스피200 선물 2026년 12월물 현재가 조회에 성공했습니다.
두 호출 모두 상태코드 200 이며, 응답 원본은 `results/` 에 있습니다.

### 겪은 오류

| 오류 | 내용 | 원인과 조치 |
| --- | --- | --- |
| `EAU024` | `appKey 발급 서비스와 토큰 발급 요청 서비스가 일치하지 않습니다` (HTTP 400) | 실전 계좌로 발급한 키를 모의 서버에 보냈기 때문. 모의계좌로 API 를 다시 신청해 해결 |
| `5012` | HTS 모의거래 로그인 시 `사용자ID 미존재` | 모의거래는 HTS 아이디 발급일 기준 **다음 영업일부터** 신청 가능. 계좌 개설 당일에는 로그인되지 않음 |

APP Key 와 Secret Key 는 모의·실전 환경별로, 국내·해외 시장별로 각각 발급됩니다.

### 다음 주차 준비

현재가 조회는 시점 하나의 값만 주므로, 지표 계산과 백테스트에는 시계열이 필요합니다.
2주차에는 분봉 조회(`/trade/v1/domestic/chart-minute`)를 붙일 예정입니다.

---

APP Key·Secret Key 와 호출 응답의 인증정보는 저장소에 올리지 않았습니다.

[전체 일정으로](../README.md)
