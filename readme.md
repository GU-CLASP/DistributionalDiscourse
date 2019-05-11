# To Do

- text preprocessing
  - possibly eliminate things inside << >> (e.g., <<sounds like she's washing dishes>>)
  - possibly eliminate things inside < > (e.g., <noise>; <baby_crying>)
- loading word vectors 
  - reset embedding weights for tokens that are in the word vector vocab but have a specific annotation meaning ('[', /', --', '+', etc.)
- model training
  - it seems that there's a batch about half way through that always has really high loss. What's with that?
  - check calculation for "Current loss" during training
  - compute & report validation loss/accuracy after each epoch.
  - save trained model??
  - experiment with optimizer hyperparameters?
  - train_encoder flag should really be train_embedding (we do want higher level layers in wordvec models trained even when the embedding isn't)
