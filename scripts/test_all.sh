#!/bin/bash
echo "Running copy for NTM"
python main.py --mode 2 --config 1 --model copy-ntm > data/copy.timing &
echo "Running repeat copy for NTM"
python main.py --mode 2 --config 2 --model repeat-copy-ntm > data/repeat-copy.timing &
echo "Running associative for NTM"
python main.py --mode 2 --config 3 --model associative-ntm > data/associative.timing &
