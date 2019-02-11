#!/bin/bash

for d in model*/ ; do
    echo "$d"
    cd $d
    cp ../../scripts/steps_all.sh .
    cp ../../scripts/step2_train_model.sh .
    cp ../../scripts/step3_pack_model.sh .
    cp ../../scripts/test_performance_model.py .
    cp ../run_model_allarticles.py .
    #cp -r ../../myallennlp .
    cd ..
done
