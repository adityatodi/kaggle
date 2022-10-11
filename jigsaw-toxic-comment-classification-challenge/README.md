# Toxic Comment Classifier

A model to classify whether a given sentence is toxic or not and what level of toxicity does it classify. 
Completed this as a part of [Kaggle Competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/leaderboard)

## Model Architecture

1. **DistilBERT** base used for Sentence Representation
2. 2 Layers of feed-forward network _(nn.Linear/Relu/Dropout)_
3. Classifier Layer (nn.Linear)

## Overview

- Converted comment text to sentence embedding using HuggingFace's pre-trained model.
- Trained 6 different models to classify whether a given comment is - Toxic, Severe `Toxic`, `Obscene`, `Insult`, `Threat` or `Identity Hate`

## Scores

`Accuracy - 73.923%` in _Kaggle Competition_

## Setup

1. Install `requirement.txt` packages
2. In `model.ipynb` uncomment codes for making input encoding.
3. Models for each classifier are already present, code to train the model is present but commented out.

## Predictions

- Encode the comment-text by using tokenizer (_present in ipynb_)
- Call the models to get the respective classification results for each model
## Future Work

1. Pipeline the end to end process
2. Make single function to get predictions from all models
3. Improve the accuracy by parameterizing the sentence representations also

