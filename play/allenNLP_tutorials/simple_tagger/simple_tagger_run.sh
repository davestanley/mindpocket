# Run the following
allennlp train ./simple_tagger.json --serialization-dir /tmp/tutorials/getting_started
allennlp evaluate /tmp/tutorials/getting_started/model.tar.gz ./sentences.small.test
