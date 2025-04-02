#!/bin/bash

# 定义参数数组
layers=(1 2 3)
window_sizes=(150000 15000 15000 1500)
stride_sizes=(150000 15000 7500 1500)

# 外层循环：遍历不同的 layers
for layer in "${layers[@]}"; do
  # 内层循环：遍历 window_sizes 与 stride_sizes 的索引
  for i in "${!window_sizes[@]}"; do
    window=${window_sizes[$i]}
    stride=${stride_sizes[$i]}
    
    # 根据参数构造文件名，这里的文件名格式请根据你的实际情况修改
    file="datasets/results/mlp_mae_test_results_w_${window}_s_${stride}_hidden_100_layers_${layer}_epochs_100_batch_32_lr_0.001_dropout_0.5.csv"
    
    # 构造提交时的信息
    message="layers=${layer}, window=${window}, stride=${stride}"
    
    # 显示当前提交信息（可选）
    echo "Submitting file: ${file}"
    echo "Message: ${message}"
    
    # 提交到 Kaggle
    kaggle competitions submit -c LANL-Earthquake-Prediction -f "${file}" -m "${message}"
    
    # 可以根据需要添加适当的延时，防止请求过快
    sleep 2
  done
done
