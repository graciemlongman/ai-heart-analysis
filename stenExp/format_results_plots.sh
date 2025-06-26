#!/bin/bash

# Activate the virtual environment
source /home/lunet/nc0051/.conda/envs/project/bin/activate

# Change to the directory containing the Python script
cd ~/PROJECT/ai-heart-analysis

# Find all "results" folders
results_folders=$(find . -type d -name "results" | grep -v -E 'postproc|yolo|LSTM')

# Run the Python script
for path in $results_folders; do
    model=$(echo "$path" | cut -d'/' -f4)
    opt_candidate=$(echo "$path" | cut -d'/' -f5)

    if [ "$opt_candidate" == "one" ]; then
        optim="Adam"
    else
        optim="$opt_candidate"
    fi

    python stenExp/test.py "$model" "$optim"
done
