#!/bin/bash

DIRECTORY=model_params

for d in model*/ ; do
    echo "$d"
    cd $d
    if [ ! -d "$DIRECTORY" ]; then
       # If directory doesnt exist, run the script
      echo ./steps_all.sh
    fi
    cd ..
done

