#!/bin/bash
# Team03 week02: snapshot paper runs, end the check run at close, copy the journal. Paper runs only.
PHASE=$1; OUT=$HOME/nh_week02/auto; API=http://127.0.0.1:47100/api/strategy
TS=$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S); LOG=$OUT/automation.log
log() { echo "$(TZ=Asia/Seoul date "+%F %T") [$PHASE] $*" >> $LOG; }
save() { curl -s -o $OUT/$2_$TS.json $API$1 && log "saved $2_$TS.json ($(stat -c%s $OUT/$2_$TS.json) bytes)"; }
systemctl is-active --quiet nh-futures.service || { log "service down"; }
save /runs runs_$PHASE
save /runs/run-team03-w2-s4 snapshot_s4_$PHASE
save /runs/run-team03-w2-check snapshot_check_$PHASE
if [ "$PHASE" = close ]; then
  log "end check run -> $(curl -s -o /dev/null -w %{http_code} -X POST $API/runs/run-team03-w2-check/end)"
  sleep 3
  save /runs runs_after_end
  save /runs/run-team03-w2-check snapshot_check_ended
  for f in $HOME/NHFutures/state/strategy-trades.jsonl*; do [ -e "$f" ] && cp "$f" $OUT/${TS}_$(basename $f) && log "copied $(basename $f)"; done
fi
