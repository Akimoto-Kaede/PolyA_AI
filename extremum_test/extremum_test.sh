#!/bin/bash
#BSUB -J extremum_test
#BSUB -q normal
#BSUB -o /scem/work/zhangyt/log/output/%J.stdout
#BSUB -e /scem/work/zhangyt/log/error/%J.err
#BSUB -n 36
#BSUB -m c009
#BSUB -R "span[ptile=36]"

. /scem/work/zhangyt/anaconda3/bin/activate
conda activate tf

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python /scem/work/zhangyt/project/ai/extremum_test/pal_train_extremum.py
