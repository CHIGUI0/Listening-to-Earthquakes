#!/bin/bash

# 定义窗口大小和步长数组
window_sizes=(150000 15000 15000 1500 1500)
stride_sizes=(150000 15000 7500 1500 750)

# 模型参数
HIDDEN_SIZE=100
DROPOUT=0.5

# 训练参数
NUM_EPOCHS=100
BATCH_SIZE=32
LR=0.001
WEIGHT_DECAY=1e-4

# 循环不同窗口配置和不同 LSTM 层数
for i in {0..4}
do
  WINDOW_SIZE=${window_sizes[i]}
  WINDOW_STRIDE=${stride_sizes[i]}

  TRAIN_FILE_PATH="datasets/final_npz/LSTM_top_5/train_features_w_${WINDOW_SIZE}_s_${WINDOW_STRIDE}.npz"
  TEST_FILE_PATH="datasets/final_npz/LSTM_top_5/test_features_w_${WINDOW_SIZE}_s_${WINDOW_STRIDE}.npz"

  for NUM_LAYERS in 1 2 3
  do
    echo "Experiment：WINDOW_SIZE=${WINDOW_SIZE}, WINDOW_STRIDE=${WINDOW_STRIDE}, NUM_LAYERS=${NUM_LAYERS}"

    python LSTM.py \
      --train_file_path "$TRAIN_FILE_PATH" \
      --test_file_path "$TEST_FILE_PATH" \
      --window_size "$WINDOW_SIZE" \
      --window_stride "$WINDOW_STRIDE" \
      --num_epochs "$NUM_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --hidden_size "$HIDDEN_SIZE" \
      --num_layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --weight_decay "$WEIGHT_DECAY" \
      --test
  done
done
