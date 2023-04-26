#!/usr/bin/env bash

python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 5 --scale 1 delaunay_n24/1proc/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 5 --scale 2 delaunay_n24/2proc/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 5 --scale 4 delaunay_n24/4proc/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 5 --scale 8 delaunay_n24/8proc/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 1 --scale 16 delaunay_n24/16proc_backup/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 5 --scale 32 delaunay_n24/32proc/
python extract_data.py --type dedup --scenario "Delaunay N24" --size-time-file --num-chkpts 5 --output scaling_data.csv --num-runs 2 --scale 64 delaunay_n24/64proc/

