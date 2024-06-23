**Execution**
1. python elmo.py
-- This will generate two dictionaries word2idx.pt and idx2word.pt. These dictionaries will be used further for the downstream classification task. 
-- Only 20000 sentences from the train corpus have been used for generating the dictionaries as well as for training.
-- After training, two pretrained models named forward_model.pt and backward_model.pt are generated (instead of a single bilstm.pt)

2. python classification.py
-- This will generate the classifier.pt file.

**Points to note**
1. The hyperparameter tuning part consists of three parts:
--Trainable lambdas
--Frozen lambdas
--Learnable function

All these parts are implemented in the complete.ipynb notebook.

**Drive link for pretrained models**
https://drive.google.com/drive/folders/1nJ_U2YxAkElYL9Ps0FoA9zkWmiXXzdyB