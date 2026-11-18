#!/usr/bin/env bash
# Fetch the authors' code (MIT, github.com/mims-harvard/patient-safety) at the pinned commit into original/, and the
# processed FAERS data + their result tables (Harvard Dataverse doi:10.7910/DVN/G9SHDA v1.0, CC0) into data/.
# Each file is checked against the MD5 Dataverse publishes; sha256 of what we got is written to checksums.sha256.
# AE_mapping.pk carries MedDRA terms (MedDRA licence: non-commercial) -> data/ is gitignored, never commit it.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=d3a62d01f980320048f97d34bd8c1cd7ee464f8f   # 2021-10-11, last commit
[ -d original ] || git clone -q https://github.com/mims-harvard/patient-safety original
git -C original checkout -q "$COMMIT"
test "$(git -C original rev-parse HEAD)" = "$COMMIT"
git -C original status --porcelain | grep -q . && { echo "original/ is modified"; exit 1; } || true
mkdir -p data
API=https://dataverse.harvard.edu/api/access/datafile
# dataverse-id  md5  name  (only the files the notebooks read, plus the authors' result tables; CSV copies of the
# inputs, drug_ATC_mapping.csv and XML_NTS.pdf are not used by steps 3-7 and are skipped)
while read -r id md5 name; do
  f="data/$name"
  [ -s "$f" ] || curl -fsSL --retry 5 -o "$f" "$API/$id"
  echo "$md5  $f" | md5sum -c --quiet || { echo "MD5 mismatch: $f"; exit 1; }
done <<'LIST'
4416415 2b4005a36af8a7da12d71c980951ed9d patient_safety.pk
4416416 a49edd2fedb8a15f200a999c5a6f079a AE_dic.pk
4416414 480a01b631d8d45b5e3850186c82f461 AE_mapping.pk
4416417 84ee0adb760887b56e442b1320e9b228 drug_mapping.pk
4416446 faeacff798d6119a7455a6821e836b4a Identified_adverse_events_in_all_patients.csv
4416447 0a6407f1ca307d92fd1884cd675ee1c0 Identified_adverse_events_in_male.csv
4416443 0e7a356d6fdf5793c49459bbaf26d733 Identified_adverse_events_in_female.csv
4416444 527f809861cd2a910d3d18a8dc32b858 Identified_adverse_events_in_young .csv
4416448 b96130352f106aff3695646d3e818798 Identified_adverse_events_in_adults.csv
4416445 f3f38a2f44c6ad2093ae41c6f9182eac Identified_adverse_events_in_elderly.csv
4416450 ebe0be0892b3d92a5dea1ec3c1dfa3ef Associations_between_adverse_events_and_the_pandemic.csv
4416451 ead03acf3695a4043ce42a0953c4908a Associations_between_adverse_events_and_the_pandemic_Top100.csv
4416454 ae7dbeef5381f2b645c4b4bf13032600 Associations_between_adverse_events_and_drugs.csv
4416452 979746647d029ee6ddaec7130401c05c Associations_between_adverse_events_and_drugs_Top100.csv
4416455 dc04e50f761ce3409b1ffb22a9f0d5d0 Associations_between_drug_adverse_event_pairs_and_the_pandemic.csv
4416449 97a774a271a6cc94a0523a226d263bab Associations_between_drug_adverse_event_pairs_and_the_pandemic_Top100.csv
LIST
if [ -f checksums.sha256 ]; then sha256sum -c --quiet checksums.sha256; else sha256sum data/* > checksums.sha256; echo "wrote checksums.sha256 (commit it)"; fi
