#!/usr/bin/env bash

if [ "$#" -ne 4 ] || ! [ -x "$1" ] || ! [ -f "$2" ] || ! [ -f "$3" ]; then
  echo "Usage: $0 FAST_ALIGN SRC_FILE TRG_FILE OUTPUT" >&2
  exit 1
fi

fast_align=$1
src_file=$2
trg_file=$3
output=$4

filename=`mktemp`

pr -mts' ||| ' ${src_file} ${trg_file} >> ${filename}
${fast_align} -i ${filename} -d -o -v -r > ${output} #2>/dev/null
rm -f ${filename}

