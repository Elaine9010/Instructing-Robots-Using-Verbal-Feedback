#!/bin/bash

# Specify the script to be executed
script_to_run="evaluation_B.py"

# Specify the parameters you want to use
parameters=(
    "7B 0 seen seen"
    "7B 0 seen unseen"
    "7B 0 unseen unseen"
)

# Loop through the parameters and run the script
for param in "${parameters[@]}"; do
    echo "Running script with parameters: $param"
    python "$script_to_run" $param

done
