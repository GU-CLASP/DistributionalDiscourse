Sun et al. 2019 - How to fine-tune BERT for text classification 
- Data preprocessing in data section
  - word piece tokenization
  - vocabulary size
  - disfluencies
  - laughter normalization?
- Hyperparameters
  - addl pre-training / fine-tuniing
  - batch size
  - max sequence length
  - learning rate / warm-up 
  - optimizer / settings
  - dropout
- justification for using the last layer (§5.2.3)
- learning rate (2e-5) is for catastrophic forgetting; (4e-4) fails to converge
- finds that within task pre-training is sometimes helpful (like our combined corpus)

Peters et al. 2019 - To tune or not to tune
- fine-tuning is better when the pre-training and target tasks are very different
- mutual information between layers of fine-tuned & pre-trained model 
  - https://github.com/mrtnoshad/EDGE

- reference for combining next sentence & masked token?
- SWDA -> SwDA
- table -> Table; figure -> Figure
- consistent table \axes
- subfigure box plots
-
