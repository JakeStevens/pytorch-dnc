#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  exit
fi

arg=$1
task=${arg%????}
if [ $task = "copy" ]; then
  config=1
elif [ $task = "repeat-copy" ]; then
  config=2
elif [ $task = "associative" ]; then
  config=3
elif [ $task = "ngrams" ]; then
  config=4
elif [ $task = "priority-sort" ]; then
  config=5
else
  echo "Error: Please use a supported task"
  exit
fi

echo "Running with 120 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 120 > data/$1_120.timing
echo "Running with 256 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 256 > data/$1_256.timing
echo "Running with 512 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 512 > data/$1_512.timing
echo "Running with 1024 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 1024 > data/$1_1024.timing
echo "Running with 2048 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 2048 > data/$1_2048.timing
echo "Running with 4096 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 4096 > data/$1_4096.timing
echo "Running with 8192 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 8192 > data/$1_8192.timing
echo "Running with 16384 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 16384 > data/$1_16384.timing
echo "Running with 32768 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 32768 > data/$1_32768.timing
echo "Running with 65536 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 65536 > data/$1_65536.timing
echo "Running with 131072 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 131072 > data/$1_131072.timing
echo "Running with 262144 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 262144 > data/$1_262144.timing 
echo "Running with 524288 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 524288 > data/$1_524288.timing
echo "Running with 1048576 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 1048576 > data/$1_1048576.timing
echo "Running with 2097152 slots"
python main.py --mode 2 --config $config --model $1 --mem_slots 2097152 > data/$1_2097152.timing
