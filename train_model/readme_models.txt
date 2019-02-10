Model naming convention:
modelWe_X.aYZ
  W - model-type / use weighted entropy 
    0 - LSTM / no
    1 - LSTM / yes
    2 - RNN / no
    3 - RNN / yes
    4 - Linear / no
    5 - Linear yes
  X - model complexity (character embedding)
    0 - Word embedding, no character embedding
    1 - Word embedding, character embedding
    2 - No word embedding, no character embedding
  a - Word embedding settings
    a - Standard embedding (50 units)
    b - GLoVE50 with trainable = true
    c - GLoVE50 with trainable = false
    d - GLoVE100 with trainable = false
  Y - Tag embedding settings
    1 - 
    2 - Standard network size (default character, tag embedding)
          pos_coarse_tags = false; pos_embedding = 5; ner_embedding = 7; dependency_label = 10
    3 - Same as 2, but with pos_coarse_tags
          pos_coarse_tags = true; pos_embedding = 5; ner_embedding = 7; dependency_label = 10
    4 - Same as 3, but with fuller embedding
          pos_coarse_tags = true; pos_embedding = 15; ner_embedding = 19; dependency_label = 20
  Z - Parts of speech tagging included (Pos, Dep, Ner)
    0 - 0,0,0
    1 - 0,0,1
    2 - 0,1,0
    3 - 0,1,1
    4 - 1,0,0
    5 - 1,0,1
    6 - 1,1,0
    7 - 1,1,1



Notes on vocab sizes:
# Results of survey of sizes. Use these for embedding:
# (Pdb) Npos 15: (coarse tags)
# (Pdb) Npos 49: (fine tags)
# (Pdb) Nner 19:
# (Pdb) Ndep 45:

