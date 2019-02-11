#!/bin/bash

# All steps necessary to run model

# Activate allenNLP
source activate allennlp

# Train the model and pack the model too
DIRECTORY=model_params
if [ ! -d "$DIRECTORY" ]; then
  # If directory doesnt exist, run the script
  ./step2_train_model.sh
  ./step3_pack_model.sh
  mkdir model_logs
  cp model_params/*.log model_logs/
fi

# Run the model on our data
python run_model_allarticles.py

# Finally, test the model's performance
python test_performance_model.py
