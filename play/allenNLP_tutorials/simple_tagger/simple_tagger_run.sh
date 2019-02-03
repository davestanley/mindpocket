# Run the following
allennlp train ./simple_tagger.json --serialization-dir ./model_params
allennlp evaluate ./model_params/model.tar.gz ./sentences.small.test
