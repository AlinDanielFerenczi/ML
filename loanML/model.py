from keras import Sequential
from keras.layers import Dense, Dropout
import preprocessing

X_train, X_test, y_train, y_test = preprocessing.preprocess()

model = Sequential()

model.add(Dense(units=12, activation='relu', input_shape=(9, )))
model.add(Dropout(0.5))
model.add(Dense(units=6, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=64, batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)

print('test loss, test acc:', score)
