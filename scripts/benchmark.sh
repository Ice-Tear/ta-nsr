#!/bin/bash
scan_list=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
for scan in ${scan_list[*]}
do
  echo $scan;
  starttime=`date +'%Y-%m-%d %H:%M:%S'`
  python launch.py --config configs/base.yaml --train --case dtu_scan$scan
  endtime=`date +'%Y-%m-%d %H:%M:%S'`
  start_seconds=$(date --date="$starttime" +%s);
  end_seconds=$(date --date="$endtime" +%s);
  echo "dtu_scan$scan的运行时间： "$((end_seconds-start_seconds))"s" >> training_time.txt
done