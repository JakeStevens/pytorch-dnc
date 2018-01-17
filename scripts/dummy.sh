#!/bin/bash
arg=$1
task=${arg%????}
echo $task
if [ $task = "copy" ]; then
  echo "1"
elif [ $task = "repeat-copy" ]; then
  echo "2"
elif [ $task = "associative" ]; then
  echo "3"
elif [ $task = "ngrams" ]; then
  echo "4"
elif [ $task = "priority-sort" ]; then
  echo "5"
else
  echo "Error"
  exit
fi
echo "Do more"
