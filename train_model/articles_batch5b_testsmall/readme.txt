Datasets: Included Allen NER as candidate blanks
Also: Small dataset, with using cross-entropy


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
    2 - Standard network sizes (minimal)
    3 - Larger network (approaching those used for example CRF NER tagger)
