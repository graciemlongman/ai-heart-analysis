#!/bin/bash

# Change to the directory containing the Python script
cd ~/PROJECT/ai-heart-analysis

# Find all "results" folders
#results_folders=$(find . -type d -name "results" | grep -v -E 'postproc|yolo|LSTM')
results_folders=$(find . -type d -name "one" | grep -v -E 'yolo|LSTM')

# Run the Python script
for path in $results_folders; do
    
    model=$(echo "$path" | cut -d'/' -f4)
    echo "$model"
    python stenExp/test.py "$model" #"$optim"
done
