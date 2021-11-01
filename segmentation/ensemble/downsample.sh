#!/bin/bash
NUMBERS=$(seq 1 13)
DATE=20210613_4000
for NUM in ${NUMBERS}
do
    sbatch scripts/${DATE}/submit${NUM}.sh
    sleep 1
    echo "Done ${NUM}"
done
