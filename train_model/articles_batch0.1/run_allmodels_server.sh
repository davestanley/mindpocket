#!/bin/bash

DIRECTORY=model_params

for d in model*/ ; do
    echo "$d"
    cd $d
    echo Running model in $d
    ./steps_all.sh
    cd ..
    git add *
    git commit -m "Server auto commit"
done
