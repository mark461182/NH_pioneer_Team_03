#!/bin/bash
# 매주 금 16:30 KST: 앱 시뮬레이션 기록을 정리해 Team_03 저장소 main 에 올린다 (deploy key: github-nh-team03).
set -u
REPO=$HOME/nh-team03-repo; LOG=$HOME/nh_week02/auto/live_push.log
log() { echo "$(TZ=Asia/Seoul date "+%F %T") $*" >> $LOG; }
mkdir -p $(dirname $LOG)
if [ ! -d $REPO/.git ]; then git clone -q git@github-nh-team03:mark461182/NH_pioneer_Team_03.git $REPO || { log "clone failed"; exit 1; }; fi
cd $REPO
git config user.name "풀옵션 자동정리 (jarvis-brain)"; git config user.email "263942265+mark461182@users.noreply.github.com"
git checkout -q main && git pull -q --rebase origin main || { log "pull failed"; exit 1; }
WEEK=$(TZ=Asia/Seoul python3 -c "from datetime import datetime,timezone,timedelta;y,w,_=datetime.now(timezone(timedelta(hours=9))).isocalendar();print(f\"{y}-W{w:02d}\")")
python3 live/weekly_report.py --week $WEEK >> $LOG 2>&1 || { log "report failed"; exit 1; }
git add live
if git diff --cached --quiet; then log "no change"; exit 0; fi
git commit -q -m "[자동] $WEEK 앱 시뮬레이션 실행 기록 정리" && git push -q origin main && log "pushed $WEEK" || log "push failed"
