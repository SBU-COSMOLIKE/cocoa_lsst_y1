#!/bin/bash
# Assuming cocoa environment is prepared
# And pwd is cocoa/Cocoa

for i in {2..29}; do
    cobaya-run ./projects/lsst_y1/PCAs/yamls/COLA_${i}.yaml -f
    cobaya-run ./projects/lsst_y1/PCAs/yamls/EE2_${i}.yaml -f
done