import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import SGD
#%%
digit_df = pd.read_csv('digit-recognizer/train.csv')
X_test = pd.read_csv('digit-recognizer/test.csv')
y_test = pd.read_csv('digit-recognizer/sample_submission.csv')
X = digit_df.drop('label', axis = 1)
y = digit_df.label
#%% Processing Features
X = X.to_numpy()
X_test = X_test.to_numpy()
X = X.reshape(X.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#%%
X.astype('float32')
X_test.astype('float32')
X = X/255.0
X_test = X_test/255.0
#%% Processing Label
n_classes = 10
y = np_utils.to_categorical(y, n_classes)
#%%
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#%%
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid), verbose=1)
#%%
y_pred = model.predict_classes(X_test)
imageId = list(range(1, 28001))
#%%
digit_pred = pd.DataFrame(list(zip(imageId, y_pred)),columns =['ImageId', 'Label'])
digit_pred.to_csv("digit_prediction.csv", index = False)