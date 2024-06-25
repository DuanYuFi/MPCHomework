#!/bin/bash

# 定义一个函数来终止所有子进程
terminate_processes() {
  echo "Terminating all processes..."
  kill $PID0 $PID1 $PID2
}

# 捕获 SIGINT 信号并调用 terminate_processes 函数
trap terminate_processes SIGINT

# 启动 python $1 0 并获取其进程ID
python "$1" 0 &
PID0=$!

# 启动 python $1 1 并获取其进程ID
python "$1" 1 > player_1.log 2>&1 &
PID1=$!

# 启动 python $1 2 并获取其进程ID
python "$1" 2 > player_2.log 2>&1 &
PID2=$!

# 等待所有进程结束
wait $PID0
wait $PID1
wait $PID2