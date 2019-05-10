import unittest
import random
from train import train
import model

# synthetic training data
random.seed(777)
input_vocab = list(range(120)) # 120 token vocab
label_vocab = list(range(17))  # 17 DA tags
data = [[random.choices(input_vocab, k=random.randint(3,10)) # utts of 3-10 tokens
        for r in range(random.randint(2,15))] # 2-15 utts per dialogue
        for i in range(100)] # 100 dialogues
labels = [random.choices(label_vocab, k=len(convo)) for convo in data]

class TestTraining(unittest.TestCase):

    def test_word2vecavg(self):
        utt_dims = 200
        n_hidden = 250
        vocab_size = len(input_vocab)
        n_labels = len(label_vocab)
        train_data = list(zip(data, labels))
        epochs = 5
        batch_size = 10
        utt_encoder_model = model.WordVecAvg.random_init(vocab_size, utt_dims)
        dar_model = model.DARRNN(utt_dims, n_labels, n_hidden, 1, dropout=0)
        print("Testing Word2VecAvg on random inputs.")
        train(utt_encoder_model, dar_model, train_data, 
                epochs, batch_size, len(label_vocab), train_encoder=True)

