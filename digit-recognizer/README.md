# Digit Recognizer Model
## Model
```
Convolution Neural Network
- three layer Conv2D with filter size of (3 x 3) with activation function - ReLu
- a hidden layer with 100 neurons
- output layer with 10 neruons - each coinciding with one digit, activation function - Softmax
```

## Data Details
```
The image was 28 * 28 pixels.
```
## Scores
```
Accuracy - 98.8
```
## To Use the Model

Use pickle package to load the model
```python
loaded_model = pickle.load(open('digit-recognizer.model', 'rb'))
```
