#!/usr/bin bash

for entry in "$1"/*
do
  echo "$entry"
  cat -n $entry | sort -uk2 | sort -nk1 | cut -f2- > temp.txt
  mv temp.txt $entry
#  sed -i -e '3d;5d' $entry
done
