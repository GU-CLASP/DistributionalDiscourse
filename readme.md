# To Do

- additional experimental conditions: 
  - discourse markers / discourse + laughter
  - freezing the utterance encoder
  - in-domain pre-training for BERT
  - GloVe aggregation for utterances
    - BiLSTM 
    - CNN / average pool?
- methodological improvements: 
  - use customized BERT vocab/word-piece tokenization for baseline models as well as BERT
- additional corpora:
  - [AMI](http://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) -- has dialogue acts 
  - [SBCSAE](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus#Intro)  -- normalize laughter & use for pre-training
- improve reporting and analysis:
  - macro F1 / macro precision? See: [Guillou et. al., 2016](https://www.aclweb.org/anthology/W16-2345) (thanks, Sharid!)
  - majority class baseline 
  - time to train / number of parameters / task-trained parameters
  - laughter impact
    - story for laughter in each of the DAs
    - which pairs of DAs is laughter most helpful at distinguishing (i.e. in the confusion matrix, where is there the biggest decrease from NL -> L?)
    - counts for DAs (total, with/wo laughs)
- not super exciting but maybe we should try:
  - DAR model hyperparameter tuning (`hidden_size`, `n_layers`, `dropout`, `use_lstm`)
  - play with learning rate 
  - use the [BERT Adam optimiser](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/optimization.py#L183) (implements a warm-up)
- probably future work: 
  - probing tasks of the hidden layer
    - predict dialogue end (or turns to end)
    - predict turn change
  - dialogue model pre-training
    - instead of training the dialogue model to predict DAs directly, predict the encoder representation of the next utterance (unsupervised)
    - test/probe by guessing DAs (or other discourse properties) with an additional linear layer

# Experiments 

## Experimental parameters

- Encoder architecture
  - BERT ? [12 layers, 24 layers]
    - ? DistilBert
    - ? RoBERTa
  - bi-LSTM [hidden size, n layers]
- DA model architecture
  - Elman RNN [hidden size]
  - LSTM [hidden size]
- Encoder pre-training
  - Pre-trained BERT (out-of-domain)
  - Randomly initialised
  - In-domain fine-tuning
    - Corpora {SWBD-SWDA, AMI-NODA}
    - Fine-tuning task {Masked token prediction, adjacent utterance}
- Learning rate (for each task)
  - N epochs (for each task)
- BERT output
  - CLS token [final layer, second-to-last layer]
  - Token average [final layer, second-to-last layer]
  - ? Token attention [final layer, second-to-last layer]
- Task-level training (fine-tuning)
  - Learning rate
  - N epochs

## Questions to answer

- Which pre-training tasks are most helpful for DA pred.?
  - Masked LM
  - Adjacent utterances
  - Both
- Does fine-tuning learn/make use of dialogue-specific features?
  - Fine-tune pre-trained BERT w/laughs (train frozen/unfrozen)
  - Fine-tune pre-trained BERT w/o laughs (train frozen/unfrozen)
- Is catastrophic forgetting happening (are we actually using the pre-training)
  - Fine-tune pre-trained BERT w/laughs (train frozen/unfrozen)
  - Fine-tune randomly initialised BERT w/laughs (train frozen/unfrozen
