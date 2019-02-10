
# Make folder and copy necessary files over
mkdir model
cp model_params/best.th model/weights.th
cp model_params/config.json model/
cp -r model_params/vocabulary model/

# Zip the model (make sure just zipping the files, not the folder)
cd model
tar -cvzf  model.tar.gz *
cd ..

# Copy zip file back here and delete the directory
cp model/model.tar.gz .
rm -rf model

