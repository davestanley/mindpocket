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
  # Remove crap from training directory
  rm model_params/model_state_epoch*.th
  rm model_params/training_state_epoch*.th

  # Force add a script in model_params, so the folder will exist, meaning we won't re-run it
  git add -f model_params/stdout.log
fi

DIRECTORY=model_logs
if [ ! -d "$DIRECTORY" ]; then
  mkdir model_logs
  cp model_params/*.log model_logs/
fi
#
# # Run the model on our data
# DIRECTORY=SQ_pp_b4m0c2
# if [ ! -d "$DIRECTORY" ]; then
#   python run_model_allarticles.py
# fi
#
# # Finally, test the model's performance
# FILE=results1_mymodel.json
# if [ ! -f "$FILE" ]; then
#   python test_performance_model.py
# fi
