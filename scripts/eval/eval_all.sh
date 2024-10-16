#!/bin/bash
scan_list=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
for scan in ${scan_list[*]}
do
  echo $scan;
  python eval_dtu.py --input_mesh Meshes/scan$scan.ply
done
