#!/bin/bash
#BSUB -J fine_tune
#BSUB -gpu "num=1"
#BSUB -q normal
#BSUB -o /scem/work/zhangyt/log/output/%J.stdout
#BSUB -e /scem/work/zhangyt/log/error/%J.err
#BSUB -n 36
#BSUB -m c035
#BSUB -R "span[ptile=36]"

. /scem/work/zhangyt/anaconda3/bin/activate
conda activate dnabert2
python /scem/work/zhangyt/project/ai/dnabert2/polya_single_read/dnabert2_gene_200.py
