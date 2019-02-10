#!bash

# All steps necessary to run model
source activate allennlp
./step2_train_model.sh
./step3_pack_model.sh
python run_model_allarticles.py

