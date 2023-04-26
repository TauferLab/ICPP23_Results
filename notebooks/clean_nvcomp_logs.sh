#!/usr/bin bash

search_dir=/the/path/to/base/dir
for entry in "$1"/*
do
  echo "$entry"
  sed -i -e '3d;5d' $entry
done
