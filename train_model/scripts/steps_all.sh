#!/bin/bash

# All steps necessary to run model
source activate allennlp
./step2_train_model.sh
./step3_pack_model.sh

# Copy stdout logs over for safe keeping
mkdir model_logs
cp model_params/*.log model_logs/
#git add -f model_logs/*.log

# Finally, run the model on our data
python run_model_allarticles.py

