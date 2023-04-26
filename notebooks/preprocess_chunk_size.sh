#!/usr/bin/ bash

python extract_data.py --type dedup --output asia_osm_vary_chunk_size.csv \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/32 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/64 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/128 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/256 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/512 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/1024 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/2048 \
  ../data/raw_logs/asia_osm/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output hugebubbles_vary_chunk_size.csv \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/32 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/64 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/128 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/256 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/512 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/1024 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/2048 \
  ../data/raw_logs/hugebubbles/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output message_race_vary_chunk_size.csv \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/32 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/64 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/128 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/256 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/512 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/1024 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/2048 \
  ../data/raw_logs/message_race/vary_chunk_size/dedup/4096 

python extract_data.py --type dedup --output unstructured_mesh_vary_chunk_size.csv \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/32 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/64 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/128 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/256 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/512 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/1024 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/2048 \
  ../data/raw_logs/unstructured_mesh/vary_chunk_size/dedup/4096 
