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
    - Corpora {SWBD-SWDA, AMI-NODA, Ubuntu}
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

### Is pre-trained BERT useful for dialogue at all?

- baseline LSTM model (1) vs. pre-trained BERT (2)
- pre-trained BERT (2) vs. randomly-initalized BERT (4) vs. BERT w/ additional pre-training (5)
- **Analysis**: 
  - Compare performance on SWDA & AMI-DA of model w/ highest val. accuracy after 20 epochs
  - Does randomly initialized BERT catch up to pre-trained BERT in performance (if so, maybe catastrophic forgetting is happening)

### Does the model make use of dialogue-specific features?

- BERT fine-tuned on SWDA/AMI-DA (2.1/2.3) vs. BERT fine tuned on SWDA-NL/AMI-DA-NL (2.2/2.4)
- LSTM trained on SWDA/AMI-DA (1.1/1.3) vs. LSTM trained on SWDA-NL/AMI-DA-NL (1.2/1.4)
- **Analysis**: 
  - Compare performance of models trained with and without laughter. If models with laughter do better, the model must be using the laughter.
  - Compare increase (assuming there is one) in performance for LSTM vs. BERT
  - DA-specific performance. Which DAs does laughter help disambiguate?
  - Performance difference on utterances with/following/preceding laughter

- BERT fine-tuned on SWDA
- **Analysis**: Compare the model trained with laughter on the test set with/without laughter. _Is this better/worse than comparing the with the model that was also trained with the no-laughter condition? Not sure..._

### Does additional in-domain pre-training help?

- BERT with additional pre-training (masked token/next utterance/both) and frozen during fine-tuning (5) vs. BERT with no additional pre-training (frozen/not-frozen) (2,6)
- **Analysis**
  - Compare performance of 5 and 2
  - How long does it take 6 to catch up to 5? (if at all)

### (bonus) How well do dialogue-tuned models transfer to other dialogue settings?

- BERT pre-trained on one domain (AMI/SWBD) and fine tuned on the other (AMI-DA/SWDA)
- BERT pre-trained on a big dialogue corpus like Ubuntu
- **Analysis** 
  - Compare performance of 5.1-3 vs. 5.4-6 on AMI and visa versa on SWDA.
  - Compare performance of 7 to 5 and 2. (Is Ubuntu pre-training as good as in-genre pre-training? Is it better than no dialogue pre-training at all?)
