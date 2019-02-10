Model naming convention:
modelWe_X.aYZ
  W - use weighted entropy
    0 - no 
    1 - yes, weight entropy
  X - model complexity (character embedding)
    0 - Word embedding, no character embedding
    1 - Word embedding, character embedding
    2 - No word embedding, no character embedding
  a - Word embedding settings
    a - Standard embedding
    b - GLoVE50 with trainable = true
    c - GLoVE50 with trainable = false
    d - GLoVE100 with trainable = false
  Y - Network size
    2 - Standard network sizes (minimal, 50 units word embedding, GLoVe 50)
                 - This means a network with word embeddding + char
                      embedding will have 100 units total
                 - A network with just word will have 50 total
    3 - Larger network (approaching those used for example CRF NER tagger)
    4 - Large network sizes (minimal, 100 units word embedding, GLoVe 100)
    No char embedding; Yes word embedding; 
      2 - has parts of speech embedding
  Z - Parts of speech tagging (Pos, Dep, Ner)
    0 - 0,0,0
    1 - 0,0,1
    2 - 0,1,0
    3 - 0,1,1
    4 - 1,0,0
    5 - 1,0,1
    6 - 1,1,0
    7 - 1,1,1





