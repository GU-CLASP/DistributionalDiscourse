General comments

# Review 1.
- Although we are only concerned with text in this work, we agree that text is not enough to capture meaning of laughter. These experiments investigate how this model deals with dialogue, given that it was pre-trained on text data. Giving the model acoustic information would undermine that goal.
- We intentionally use a very simple model, which is partly why the description is so short. We will make our code available when the work is no longer under anonymity restrictions.
- DAR is not as appropriate as NLU for laughter?? 
- fix:
  - change to the 1st edition of Austin
  - I also wouldn't say that conditional random fields are more
    "sophisticated" than most flavors of RNNs. -- clarify
- reformulate the contributions
- laughter in non-dialogue?
- why single symbol, fix in Data (ref ginzburg/chiara) but we have
  just a single type.
- why laughter is helpful? restate that in the discussion

idea: can laughter help with repair? 
# Review 2.
- Yes -- $s^t$ is the speaker token and $w_n$ is the n'th word token. Thanks for noticing this omission. 
- This is indeed an interesting discrepancy between the effect of laughter on the CNN and BERT models. We don't have a hypothesis at the moment, but we plan to investigate further. Looking at the DA tag-specific performance may provide a clue.
- The opening two paragraphs get off on the wrong foot for me. The
  first sentence is too often repeated in papers like this, and the
  rest of these two paragraphs seems to under-state the extent to
  which there is prior work on using neural models for dialogue.
  - try to re-write first two para, more laughter and multi-modal dialogue
- label outliers on Fig 1
- say that non-verbal is not accounted for in the Task 1
- Fig 3. Plot: P(qw|qh) - P(qw) instead
- numbers for qh|qw improvement
- Fig 2: utterance encoder, context representation, softmax
- Table 6, add F1
- Experiment 2 -> add L/NL condition
- Experiment 3 -> as well add laughters

# Review 3.
- Thank you for your questions suggesting avenues for deeper analysis.
- Please see figure 4 in the appendix for the affect of laughter on performance for particular dialogue acts.
- We discuss linguistic work on the functional role of laughter and its relationship to our results in section 4.1.
- say something about the split. How is it different from the
  "canonical split" (optional)
- CNN: "a simple and popular baseline for text classification or
  encoding texts"

# Review 4.
- Thank you for these constructive comments.
