#!/bin/bash
set -e

CKPT_DIR="$(dirname "$0")/checkpoints"
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$CKPT_DIR"
mkdir -p "$DATA_DIR"

echo "Downloading checkpoints to $CKPT_DIR ..."

curl -L -o "$CKPT_DIR/checkpoint_mad.pt" \
  "https://www.dropbox.com/scl/fi/dkn37jvmxdk2occ71286r/checkpoint_mad.pt?rlkey=hjzqg1hk6n83bd4hzrxwg730b&st=qiyo05te&dl=1"

curl -L -o "$CKPT_DIR/checkpoint_microenvironment.pt" \
  "https://www.dropbox.com/scl/fi/7otduegyj3croqbkysn0y/checkpoint_microenvironment.pt?rlkey=xvgf4ul7kp3sgi7zoli6ctqw6&st=qdleoz3h&dl=1"

curl -L -o "$CKPT_DIR/checkpoint_morphology.pt" \
  "https://www.dropbox.com/scl/fi/k0mqgeoso3j7us82ljs78/checkpoint_morphology.pt?rlkey=3qdvtjari17h8mhv1bjhxgib3&st=x98camgr&dl=1"

curl -L -o "$DATA_DIR/subset.h5" \
  "https://www.dropbox.com/scl/fi/cv4itgzoi3o8up032k7xl/subset.h5?rlkey=pejfsckhhpgyo7o8f5h3piim8&st=ell78e0i&dl=1"

curl -L -o "$DATA_DIR/labels.parquet" \
  "https://www.dropbox.com/scl/fi/hwfkog1uwyoo08p3mqsp2/labels.parquet?rlkey=4lzmlzpb46drg268cga0kh3u1&st=lbsnnza0&dl=1"

echo "Done. Checkpoints saved to $CKPT_DIR and data saved to $DATA_DIR"
