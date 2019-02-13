#!/bin/bash

FILE=test_performance_model.ipynb

cp ../scripts/run_allmodels.sh .
cp ../scripts/run_allmodels_server.sh .
for d in *model*/ ; do
    echo "$d"
    cd $d
    cp ../../scripts/steps_all.sh .
    cp ../../scripts/steps_all_server.sh .
    cp ../../scripts/step2_train_model.sh .
    cp ../../scripts/step3_pack_model.sh .
    cp ../../scripts/test_performance_model.py .

    # Only copy over Jupyter results journal if it's not already present (dangerous to overwrite this!)
    if [ ! -f "$FILE" ]; then
      cp ../../scripts/test_performance_model.ipynb .
    fi
    cp ../run_model_allarticles.py .
    #cp -r ../../myallennlp .
    cd ..
done
