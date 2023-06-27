#!/usr/bin/ bash

basepath=$1

python extract_data.py --type dedup --output asia_osm_vary_chunk_size.csv --scenario=AsiaOSM \
  $basepath/asia_osm/vary_chunk_size/dedup/32 \
  $basepath/asia_osm/vary_chunk_size/dedup/64 \
  $basepath/asia_osm/vary_chunk_size/dedup/128 \
  $basepath/asia_osm/vary_chunk_size/dedup/256 \
  $basepath/asia_osm/vary_chunk_size/dedup/512 \
  $basepath/asia_osm/vary_chunk_size/dedup/1024 \
  $basepath/asia_osm/vary_chunk_size/dedup/2048 \
  $basepath/asia_osm/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output hugebubbles_vary_chunk_size.csv --scenario=Hugebubbles \
  $basepath/hugebubbles/vary_chunk_size/dedup/32 \
  $basepath/hugebubbles/vary_chunk_size/dedup/64 \
  $basepath/hugebubbles/vary_chunk_size/dedup/128 \
  $basepath/hugebubbles/vary_chunk_size/dedup/256 \
  $basepath/hugebubbles/vary_chunk_size/dedup/512 \
  $basepath/hugebubbles/vary_chunk_size/dedup/1024 \
  $basepath/hugebubbles/vary_chunk_size/dedup/2048 \
  $basepath/hugebubbles/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output message_race_vary_chunk_size.csv --scenario="Message Race" \
  $basepath/message_race/vary_chunk_size/dedup/32 \
  $basepath/message_race/vary_chunk_size/dedup/64 \
  $basepath/message_race/vary_chunk_size/dedup/128 \
  $basepath/message_race/vary_chunk_size/dedup/256 \
  $basepath/message_race/vary_chunk_size/dedup/512 \
  $basepath/message_race/vary_chunk_size/dedup/1024 \
  $basepath/message_race/vary_chunk_size/dedup/2048 \
  $basepath/message_race/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output unstructured_mesh_vary_chunk_size.csv --scenario="Unstructured Mesh" \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/32 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/64 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/128 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/256 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/512 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/1024 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/2048 \
  $basepath/unstructured_mesh/vary_chunk_size/dedup/4096 
