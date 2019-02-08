Model naming convention:
modelX.aYZ
  X - model complexity (character embedding)
    0 - no character embedding (single_id word embedding)
    1 - has char embedding
    2 - Has parts of speech embedding
  a - Word embedding settings
    a - Standard embedding
    b - GLoVE50 with trainable = true
    c - GLoVE50 with trainable = false
    d - GLoVE100 with trainable = false
  Y - Network size
    2 - Standard network sizes (minimal, 100 hidden units)
    3 - Larger network (approaching those used for example CRF NER tagger)
    4 - Super large network (400 hidden units)





