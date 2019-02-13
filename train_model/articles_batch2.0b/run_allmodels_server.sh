#!/bin/bash

DIRECTORY=model_params

for d in *model*/ ; do
    echo "$d"
    cd $d
    echo Running model in $d
    ./steps_all_server.sh
    cd ..
    # Make sure everything saves!
    git add *
    git commit -m "Server auto commit"
    git push
done
