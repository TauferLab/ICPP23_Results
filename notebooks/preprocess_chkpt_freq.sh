#!/usr/bin/env bash

python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 5  \
  asia_osm/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 10 \ 
  asia_osm/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 20 \ 
  asia_osm/vary_chkpt_freq/dedup/20chkpts/64/
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 5  \ 
  asia_osm/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 10 \ 
  asia_osm/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output asia_osm_vary_chkpt_freq_64.csv --chunksize 64 --scenario AsiaOSm --num-chkpts 20 \ 
  asia_osm/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 5  \ 
  hugebubbles/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 10 \ 
  hugebubbles/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 20 \ 
  hugebubbles/vary_chkpt_freq/dedup/20chkpts/64/
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 5  \ 
  hugebubbles/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 10 \ 
  hugebubbles/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output hugebubbles_vary_chkpt_freq_64.csv --chunksize 64 --scenario Hugebubbles --num-chkpts 20 \ 
  hugebubbles/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 5 \
  message_race/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 10 \
  message_race/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 20 \
  message_race/vary_chkpt_freq/dedup/20chkpts/64/
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 5 \
  message_race/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 10 \
  message_race/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output message_race_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Message Race" --num-chkpts 20 \
  message_race/vary_chkpt_freq/nvcomp/20chkpts/

python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 5 unstructured_mesh/vary_chkpt_freq/dedup/5chkpts/64/
python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 10 unstructured_mesh/vary_chkpt_freq/dedup/10chkpts/64/
python extract_data.py --type dedup --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \
  --num-chkpts 20 unstructured_mesh/vary_chkpt_freq/dedup/20chkpts/64/
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \ 
  --num-chkpts 5 unstructured_mesh/vary_chkpt_freq/nvcomp/5chkpts/
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \ 
  --num-chkpts 10 unstructured_mesh/vary_chkpt_freq/nvcomp/10chkpts/
python extract_data.py --type nvcomp --output unstructured_mesh_vary_chkpt_freq_64.csv --chunksize 64 --scenario "Unstructured Mesh" \ 
  --num-chkpts 20 unstructured_mesh/vary_chkpt_freq/nvcomp/20chkpts/


cat asia_osm_vary_chkpt_freq_64.csv > vary_chkpt_freq_64.csv
tail -n +2 hugebubbles_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv
tail -n +2 message_race_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv
tail -n +2 unstructured_mesh_vary_chkpt_freq_64.csv >> vary_chkpt_freq_64.csv

