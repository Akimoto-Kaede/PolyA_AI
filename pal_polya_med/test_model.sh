#!/bin/bash
#BSUB -J test_model
#BSUB -q normal
#BSUB -o /scem/work/zhangyt/log/output/%J.stdout
#BSUB -e /scem/work/zhangyt/log/error/%J.err
#BSUB -n 36
#BSUB -m c008
#BSUB -R "span[ptile=36]"

. /scem/work/zhangyt/anaconda3/bin/activate
module load cuda/11.8
conda activate tf
python /scem/work/zhangyt/project/ai/pal_polya_med/pal_test.py
