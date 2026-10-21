#!/bin/bash
#BSUB -J ath_rgt
#BSUB -q normal
#BSUB -o /scem/work/zhangyt/log/output/%J.stdout
#BSUB -e /scem/work/zhangyt/log/error/%J.err
#BSUB -n 36
#BSUB -m c009
#BSUB -R "span[ptile=36]"

. /scem/work/zhangyt/anaconda3/bin/activate
conda activate tf
python /scem/work/zhangyt/project/ai/polya_dist/ath/ath_rgt_exp.py
