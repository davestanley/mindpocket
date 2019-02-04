Overview: Modification of CRF model. Not actually using CRF; just using the network layer sizes to reflect this model.



	Model Summary:
	  - Yes character embedding (much larger work; dim=128; previous dim was 50)
	  - GLoVE word embedding (trainable = true)
		- Larger hidden layer - increased to 200 from 100



Sources:
	Modification of CRF model: https://github.com/allenai/allennlp/blob/v0.8.1/tutorials/getting_started/walk_through_allennlp/creating_a_model.md
	See model specification here: https://github.com/allenai/allennlp/blob/v0.8.1/tutorials/getting_started/walk_through_allennlp/crf_tagger.json
