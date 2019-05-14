# To Do

- text preprocessing
  - possibly eliminate things inside << >> (e.g., <<sounds like she's washing dishes>>)
  - possibly eliminate things inside < > (e.g., <noise>; <baby_crying>)
- loading word vectors 
  - reset embedding weights for tokens that are in the word vector vocab but have a specific annotation meaning ('[', /', --', '+', etc.)
  - for BERT use [never_split](https://github.com/huggingface/pytorch-pretrained-BERT/tree/3fc63f126ddf883ba9659f13ec046c3639db7b7e#berttokenizer) for annotation tokens (N.B. this might not be necessary provided we don't use BertTokenizer.tokenize and only use BertToknizer.convert_tokens_to_ids)
- model training
  - it seems that there's a batch about half way through that always has really high loss. What's with that?
  - check calculation for "Current loss" during training
  - compute & report validation loss/accuracy after each epoch.
  - save trained model??
  - experiment with optimizer hyperparameters?
  - train_encoder flag should really be train_embedding (we do want higher level layers in wordvec models trained even when the embedding isn't)
