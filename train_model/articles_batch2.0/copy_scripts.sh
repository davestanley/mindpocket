#!/bin/bash

for d in model*/ ; do
    echo "$d"
    cd $d
    cp ../../steps_all.sh .
    cp ../../step2_train_model.sh .
    cp ../../step3_pack_model.sh .
    cp ../run_model_allarticles.py .
    #cp -r ../../myallennlp .
    cd ..
done
