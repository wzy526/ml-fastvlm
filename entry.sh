#!/usr/bin/env bash

TAG=$1
NODE=$2
GPUTYPE=$3
ntsctl apply -G ${GPUTYPE}=8 -n $NODE -D "`pwd`" --Alias "${TAG}" --mountTempStorage \
  --image "adas-img.nioint.com/nts/ngc-pytorch:24.05-py3" \
    -c "bash scripts/${TAG}.sh"