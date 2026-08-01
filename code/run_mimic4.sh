#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${TASK:-knn}"
ACTION="${ACTION:-train}"
DATA_DIR="${DATA_DIR:-../data/mimic4}"
CACHE_DIR="${CACHE_DIR:-preprocess_data/mimic4}"
OUTPUT_DIR="${OUTPUT_DIR:-res/mimic4/${TASK}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$TASK" in
  cluster)
    TRAIN_CACHE="$CACHE_DIR/train_cluster.pt"
    ;;
  knn)
    TRAIN_CACHE="$CACHE_DIR/train_knn.pt"
    ;;
  *)
    echo "TASK must be 'cluster' or 'knn'" >&2
    exit 2
    ;;
esac

ACTION_ARGS=()
case "$ACTION" in
  train)
    ACTION_ARGS+=(--do_train)
    ;;
  test)
    if [[ -z "${RESUME_PATH:-}" ]]; then
      echo "Set RESUME_PATH to the trained checkpoint when ACTION=test" >&2
      exit 2
    fi
    ACTION_ARGS+=(--do_test --resume_path "$RESUME_PATH")
    ;;
  *)
    echo "ACTION must be 'train' or 'test'" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" train.py \
  --dataset mimic4 \
  --data_dir "$DATA_DIR" \
  --task "$TASK" \
  --use_conv anti \
  --gcn_conv_nums 2 \
  --hidden_size 100 \
  --pair_neurons 30 \
  --hidden_dropout_prob 0.4 \
  --batch_size 256 \
  --epoch 30 \
  --learning_rate 0.0001 \
  --knn_k_values 1 3 5 \
  --selection_k 1 \
  --train_data "$TRAIN_CACHE" \
  --train_input_data "$CACHE_DIR/train_input_knn.pt" \
  --valid_cluster_data "$CACHE_DIR/valid_cluster.pt" \
  --valid_knn_data "$CACHE_DIR/valid_knn.pt" \
  --test_cluster_data "$CACHE_DIR/test_cluster.pt" \
  --test_knn_data "$CACHE_DIR/test_knn.pt" \
  --output_dir "$OUTPUT_DIR" \
  "${ACTION_ARGS[@]}"
