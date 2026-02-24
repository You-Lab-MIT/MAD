#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/datasets/hest/expression_long"
IN_DIR="/data/datasets/hest/transcripts"
SCRIPT="/home/exx/Desktop/projects/MAD/DINO/dinov2/dinov2/finetune/build_xenium_counts_long.py"

SLIDES=(
  NCBI884 NCBI883 NCBI882 NCBI881 NCBI880
  NCBI879 NCBI876 NCBI875 NCBI873 NCBI870
  NCBI867 NCBI866 NCBI865 NCBI864 NCBI861
  NCBI860 NCBI859 NCBI858 NCBI857 NCBI856
)

mkdir -p "${OUT_DIR}"

export OUT_DIR IN_DIR SCRIPT

printf "%s\n" "${SLIDES[@]}" | parallel -j 6 --eta '
  slide={}
  PYTHONPATH=./ python ${SCRIPT} \
    --in-parquet ${IN_DIR}/${slide}_transcripts.parquet \
    --slide-id ${slide} \
    --out-dir ${OUT_DIR} \
    --assigned-only
'
