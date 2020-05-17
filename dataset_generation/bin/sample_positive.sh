#!/bin/bash

POSENTFILE=$1
NEGENTFILE=$2
OUTFILENAME=$3

nlinepos=$(wc -l $POSENTFILE | awk '{print $1}')
nlineneg=$(wc -l $NEGENTFILE | awk '{print $1}')



if [ $nlinepos -ge $nlineneg ]
then
  awk -F"\\t" '{print $6}' $POSENTFILE | shuf -n $nlineneg | paste $NEGENTFILE /dev/stdin | gzip > $OUTFILENAME
else
  a='{for(i=0;i<'
  b=';i++) print $6}'
  numiter=$(($nlineneg / $nlinepos + 1))
  c="${a}${numiter}${b}"
  awk -F"\\t" "$c" $POSENTFILE | shuf -n $nlineneg | paste $NEGENTFILE /dev/stdin | gzip > $OUTFILENAME
fi
