Datasets: Included Allen NER as candidate blanks
Also: Increased the size of datasets to include ALL articles!


Model naming convention:
modelX.aYZ
  X - model complexity (character embedding)
    0 - no character embedding
    1 - has char embedding
  a - Word embedding settings
    a - Standard embedding
    b - GLoVE with trainable = true
    c - GLoVE with trainable = false
  Y - Network size
    0 - Standard network sizes (minimal)
    1 - Larger network (approaching those used for example CRF NER tagger)
