#!/bin/bash
#BSUB -J train_model
#BSUB -gpu "num=1"
#BSUB -q normal
#BSUB -o /scem/work/zhangyt/log/output/%J.stdout
#BSUB -e /scem/work/zhangyt/log/error/%J.err
#BSUB -n 36
#BSUB -m c018
#BSUB -R "span[ptile=36]"

. /scem/work/zhangyt/anaconda3/bin/activate
conda activate tf
python /scem/work/zhangyt/project/ai/single_read/ara_leaf/ath_gt_single_read.py
