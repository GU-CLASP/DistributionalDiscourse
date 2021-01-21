General comments

Review 1.
- Although we are only concerned with text in this work, we agree that text is not enough to capture meaning of laughter. These experiments investigate how this model deals with dialogue, given that it was pre-trained on text data. Giving the model acoustic information would undermine that goal.
- We intentionally use a very simple model, which is partly why the description is so short. We will make our code available when the work is no longer under anonymity restrictions.

Review 2.
- Yes -- $s^t$ is the speaker token and $w_n$ is the n'th word token. Thanks for noticing this omission. 
- This is indeed an interesting discrepancy between the effect of laughter on the CNN and BERT models. We don't have a hypothesis at the moment, but we plan to investigate further. Looking at the DA tag-specific performance may provide a clue.

Review 3.
- Thank you for your questions suggesting avenues for deeper analysis.
- Please see figure 4 in the appendix for the affect of laughter on performance for particular dialogue acts.
- We discuss linguistic work on the functional role of laughter and its relationship to our results in section 4.1.

Review 4.
- Thank you for these constructive comments.
