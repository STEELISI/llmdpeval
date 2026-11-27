#!/usr/bin/env bash
# Run the authors' R scripts in the Replication Guide's order, in a scratch copy (work/) of their package,
# inside the pinned R 3.5.1 env. Each script runs in its own Rscript process (fresh session) with its log,
# wall time and peak RSS in logs/<script>.log. Outputs (*.RData, table*.csv, figures) are copied to outputs/.
#
# Adapters (the authors' files in original/ are never edited; see RESULT.md):
#   A1 Turnout.R uses the vector SR, which only Predict.R defines (the authors ran the scripts in one session).
#      We define the same SR (and eth) before sourcing Turnout.R.
#   A2 KingEI.R line 59 selects column "Distrct" (typo for "District") and stops with "undefined columns
#      selected" after saving ei.cd.RData but before ei.prc.RData. We fix that one word in work/KingEI.R.
#   A4 Turnout.R's district loops select df$CD == d, but FL.Anon has no CD column (it is called District), so
#      unmodified they select 0 rows and every district-level observed/predicted turnout is NaN. We replace
#      $CD == d by $District == d in work/Turnout.R (5 lines, one per district loop).
#   A5 Predict.R predictions 4-5 call race.pred(party = "PID"). wru (0.0-1, and every later 0.x) then merges its
#      own pid table (columns r_pid_*) into the data, but FL.Anon already has r_pid_* columns, so merge() makes
#      r_pid_*.x / r_pid_*.y and race.pred stops: "undefined columns selected". FL.Anon's r_pid_* equal wru's pid
#      table exactly (checked: 3 PID values, same 5 numbers each). A global wrapper drops the data's r_pid_*
#      only for party= calls, then calls wru::race.pred unchanged; results are what the merge would give.
#   A3 Figure1.R computes the AUCs only for the PDF legend; after it finishes we write its AUC data.frame to
#      figure1_auc.csv so compare.py can read it.
#
# Usage: ./run.sh [all|predict|table2chain|tables|appendix]   (default all)
#        ./run.sh script <X.R>   reruns one script in the existing work/ (e.g. after a fix)
# Memory guard: each R process gets ulimit -v $VMEM_KB (default 30 GiB) so a runaway job fails instead of
# letting the OOM killer pick a neighbour's process. Parallel jobs: at most 3 R processes at once.
set -euo pipefail
cd "$(dirname "$0")"
STEP=${1:-all}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/data/mamba}
MM=${MM:-$HOME/bin/micromamba}
VMEM_KB=${VMEM_KB:-31457280}
ROOT=$(pwd)
mkdir -p logs outputs

rs() {  # rs <script.R> [R prelude] [R postlude]: run one authors' script in work/, log time + peak memory
  local s=$1 pre=${2:-} post=${3:-} t0; t0=$(date +%s)
  echo "$(date -u +%FT%TZ) start $s" | tee -a logs/run.log
  ( cd work && ulimit -v "$VMEM_KB" && /usr/bin/time -v -o "$ROOT/logs/$s.time" \
      "$MM" run -p "$ROOT/env/r351" Rscript -e "${pre}source('$s', echo = TRUE, max.deparse.length = 300); ${post}" \
  ) > "logs/$s.log" 2>&1 && st=ok || st=FAIL
  echo "$(date -u +%FT%TZ) $st $s $(( $(date +%s) - t0 ))s peakRSS=$(awk -F': ' '/Maximum resident/{printf "%.1fGB", $2/1048576}' logs/$s.time)" | tee -a logs/run.log
  [ $st = ok ]
}

if [ "$STEP" = script ]; then
  rs "$2"; exit
fi

if [ "$STEP" = all ] || [ "$STEP" = predict ]; then
  ./fetch.sh
  rm -rf work && cp -rp original work                        # -p keeps download mtimes, so outputs are the newer files
  sed -i 's/"Distrct"/"District"/' work/KingEI.R                 # A2
  sed -i 's/\$CD == d/$District == d/' work/Turnout.R             # A4
  A5='race.pred <- function(voters, ..., party) { if (missing(party)) return(wru::race.pred(voters, ...)); voters <- voters[!grepl("^r_pid_", names(voters))]; wru::race.pred(voters, ..., party = party) }; '
  rs Predict.R "$A5"                                             # df.RData, df.pred1..9.RData
fi

if [ "$STEP" = all ] || [ "$STEP" = table2chain ]; then
  SRDEF='eth <- c("whi", "bla", "his", "asi", "oth"); SR <- c("SR.WHI", "SR.BLA", "SR.HIS", "SR.ASI", "SR.OTH"); '
  ( rs Precincts.R && rs Goodman.R && rs KingEI.R ) & p1=$!
  ( rs Turnout.R "$SRDEF" ) & p2=$!                              # A1
  ( rs Table1.R; rs Figure1.R "" 'write.csv(AUC, "figure1_auc.csv")'; rs Table7.R ) & p3=$!   # A3
  wait $p1 $p2 $p3 || true
fi

if [ "$STEP" = all ] || [ "$STEP" = tables ]; then
  for s in Table2.R Table5.R Table6.R; do rs $s || true; done
fi

# Appendix: Figures 2-3 and Table 3 (guide steps 8, 9, 4). Not part of "all". Table 4 (EM, doMPI cluster) is not run.
if [ "$STEP" = appendix ]; then
  rs Residuals.R && rs Figure2.R || true
  rs Figure3.R || true
  rs L2.Demo.R && rs Predict.L2.R && rs Table3.R || true
fi

find work -maxdepth 1 -type f -newer work/Codebook.txt ! -name "*.R" -exec cp {} outputs/ \;
cat > outputs/provenance.json <<JSON
{"finished": "$(date -u +%FT%TZ)", "host": "$(hostname)", "step": "$STEP",
 "r": "$("$MM" run -p env/r351 Rscript -e 'cat(R.version.string)')",
 "note": "authors' code unchanged except adapters A1-A5; see RESULT.md"}
JSON
echo "outputs in $ROOT/outputs; per-script logs in $ROOT/logs"
