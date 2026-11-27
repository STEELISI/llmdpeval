#!/usr/bin/env bash
# Same as run.sh, but the authors' script runs section by section, each in a fresh R process (2 GB XDC).
# Resumable: a section whose checkpoint exists is skipped. Sections start at the script's own header comments.
# Sections 4-5 are the two mediate(sims=1000) calls; if the XDC cannot hold them, run those two sections
# elsewhere from checkpoint_3.RData (see RESULT.md) and resume here.
set -euo pipefail
cd "$(dirname "$0")"
./fetch.sh
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mamba}
MM=${MM:-$HOME/bin/micromamba}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
SECTIONS="1:27 28:50 51:135 136:158 159:193 194:228 229:253 254:374 375:449 450:614 615:703"
[ -f work/.sectioned ] || { rm -rf work && cp -r original work && touch work/.sectioned; }
# Resume after the highest checkpoint present (earlier ones are deleted as the run goes).
last=$(ls work/checkpoint_*.RData 2>/dev/null | sed 's/.*_\([0-9]*\)\.RData/\1/' | sort -n | tail -1); last=${last:-0}
prev=none; [ "$last" -gt 0 ] && prev=checkpoint_$last.RData; i=0
for s in $SECTIONS; do
  i=$((i+1)); ck=checkpoint_$i.RData
  [ "$i" -le "$last" ] && continue
  echo "$(date -u +%FT%TZ) start section $i (lines $s)" >> sections.log
  (cd work && "$MM" run -p ../env/r351 Rscript ../env/run_section.R Labor_White_Racial_Politics_Code.R "${s%:*}" "${s#*:}" "$prev" "$ck") >> run.log 2>&1
  echo "$(date -u +%FT%TZ) done  section $i: $(tail -1 work/sections_mem.tsv)" >> sections.log
  [ "$prev" != none ] && [ "$prev" != checkpoint_3.RData ] && rm -f "work/$prev"   # keep the pre-mediation one
  prev=$ck
done
mkdir -p outputs && find work -maxdepth 1 -type f -newer work/.sectioned ! -name '*.RData' -exec cp {} outputs/ \;
cat > outputs/provenance.json <<JSON
{"finished": "$(date -u +%FT%TZ)", "host": "$(hostname)", "mode": "sectioned (run_sections.sh)",
 "r": "$("$MM" run -p env/r351 Rscript -e 'cat(R.version.string)')",
 "note": "authors' code unchanged, run in 11 sections with workspace checkpoints; mediation/matching bootstraps are unseeded in the original"}
JSON
echo "$(date -u +%FT%TZ) ALL SECTIONS DONE" >> sections.log
