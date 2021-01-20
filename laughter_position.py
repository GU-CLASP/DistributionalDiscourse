import torch
import model
import json
from eval_model import load_models

laughter_types = ['stand alone', 'prefix', 'suffix', 'infix', 'none']

def laughter_type(utt):
    laugh = '<laughter>'
    utt = re.sub(r'[#.\-,]','',utt).strip().lower()
    if utt == laugh:
        return 'stand alone'
    elif laugh in utt:
        if utt.startswith(laugh):
            return 'prefix'
        elif utt.endswith(laugh):
            return 'suffix'
        else:
            return 'infix'
    else:
        return 'none'

class LaughterTypePredictor(model.DARRNN):
    """ 
    Model for predicting laughter type of the next utterance. 
    The architecture is exactly the same as the DARRNN, but we predict laughter types
    instead of dialogue acts.
    NOTE: The prediction scheme is slightly different since we are predicting the laughter
    type of the *next* utterance (as opposed to the DA tag of *this* utterance*. But these
    differences are in the definition of the training/evaluation loops, not them model itself.
    """

    def __init__(self, utt_size, hidden_size, n_layers, dropout=0.5, use_lstm=False):
        super().__init__(utt_size, len(laughter_types), hidden_size, n_layers, 
                dropout=dropout, use_lstm=use_lstm)
    
    @classmethod
    def from_dar_model(self, dar_model):
        """
        We also allow the model to be initialized from a pre-trained DAR model, so that we can
        test whether knowledege of the DAR task helps with laughter prediction.
        """

# 1. label the data with laughter position
# 2. fine-tune the final layer of the DAR model to instead 
#    predict laughter position of the next utternace
#    - this means creating a new model type for laughter position
#    - we should allow it to be initialized from a DAR model
# 

utt_dims = 100
dar_hidden = 100
dar_layers = 1
use_lstm = False

# model_dir = "models/AMI-DA-L_bert_2019-11-20"
model_dir = "models/SWDA-L_bert_2019-11-20"

encoder_model, dar_model = load_models(model_dir)
