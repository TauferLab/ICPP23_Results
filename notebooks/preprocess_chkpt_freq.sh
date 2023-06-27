#!/usr/bin/env bash

basepath=$1

python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 5  \
  $basepath/asia_osm/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 10 \
  $basepath/asia_osm/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 20 \
  $basepath/asia_osm/vary_chkpt_freq/dedup/20chkpts/64/
. clean_nvcomp_logs.sh $basepath/asia_osm/vary_chkpt_freq/nvcomp/5chkpts
. clean_nvcomp_logs.sh $basepath/asia_osm/vary_chkpt_freq/nvcomp/10chkpts
. clean_nvcomp_logs.sh $basepath/asia_osm/vary_chkpt_freq/nvcomp/20chkpts
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 5  \
  $basepath/asia_osm/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 10 \
  $basepath/asia_osm/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSM --num-chkpts 20 \
  $basepath/asia_osm/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 5  \
  $basepath/hugebubbles/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 10 \
  $basepath/hugebubbles/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 20 \
  $basepath/hugebubbles/vary_chkpt_freq/dedup/20chkpts/64/
. clean_nvcomp_logs.sh $basepath/hugebubbles/vary_chkpt_freq/nvcomp/5chkpts
. clean_nvcomp_logs.sh $basepath/hugebubbles/vary_chkpt_freq/nvcomp/10chkpts
. clean_nvcomp_logs.sh $basepath/hugebubbles/vary_chkpt_freq/nvcomp/20chkpts
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 5  \
  $basepath/hugebubbles/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 10 \
  $basepath/hugebubbles/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 20 \
  $basepath/hugebubbles/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 5 \
  $basepath/message_race/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 10 \
  $basepath/message_race/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 20 \
  $basepath/message_race/vary_chkpt_freq/dedup/20chkpts/64/
. clean_nvcomp_logs.sh $basepath/message_race/vary_chkpt_freq/nvcomp/5chkpts
. clean_nvcomp_logs.sh $basepath/message_race/vary_chkpt_freq/nvcomp/10chkpts
. clean_nvcomp_logs.sh $basepath/message_race/vary_chkpt_freq/nvcomp/20chkpts
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 5 \
  $basepath/message_race/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 10 \
  $basepath/message_race/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 20 \
  $basepath/message_race/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 5 $basepath/unstructured_mesh/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 10 $basepath/unstructured_mesh/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 20 $basepath/unstructured_mesh/vary_chkpt_freq/dedup/20chkpts/64/
. clean_nvcomp_logs.sh $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/5chkpts
. clean_nvcomp_logs.sh $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/10chkpts
. clean_nvcomp_logs.sh $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/20chkpts
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 5 $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 10 $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 20 $basepath/unstructured_mesh/vary_chkpt_freq/nvcomp/20chkpts/


cat asia_osm_vary_chkpt_freq_64.csv > vary_chkpt_freq_64.csv
tail -n +2 hugebubbles_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv
tail -n +2 message_race_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv
tail -n +2 unstructured_mesh_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv

