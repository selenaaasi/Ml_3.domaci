import csv
import pandas as pd
# example of tf.keras python idiom
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import pandas as pd
from matplotlib import pyplot


def testModel(trainX, testX, trainy, testy):
   # define model
   model = tf.keras.Sequential()
   model.add(layers.InputLayer((4,)))

   model.add(layers.Dense(32, activation="relu"))
   #model.add(layers.Dropout(0.3))

   model.add(layers.Dense(3, activation="softmax"))

   # compile model

   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   # #early stopping
   es_callback = EarlyStopping(monitor='val_loss', patience=3)

   # fit model
   history = model.fit(trainX, trainy, batch_size=64, validation_data=(testX, testy), epochs=500, verbose=0)

   # evaluate the model
   _, train_acc = model.evaluate(trainX, trainy, verbose=0)
   _, test_acc = model.evaluate(testX, testy, verbose=0)
   print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
   # plot loss during training
   pyplot.subplot(211)
   pyplot.title('Loss')
   pyplot.plot(history.history['loss'], label='train')
   pyplot.plot(history.history['val_loss'], label='test')
   pyplot.legend()
   # plot accuracy during training
   pyplot.subplot(212)
   pyplot.title('Accuracy')
   pyplot.plot(history.history['accuracy'], label='train')
   pyplot.plot(history.history['val_accuracy'], label='test')
   pyplot.legend()
   pyplot.show()

   return model

dataframe=pd.read_table('balance-scale.data',  sep=',')
print(dataframe)
dataSet = dataframe.copy()
X = dataframe.iloc[:,0:4]
print("X")
print(X)
Y = dataSet.iloc[:, 0]
print("Y")
print(Y)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
print("Encoded y")
print(encoded_Y)
#one_hot_encoder=OneHotEncoder()
#dummy_Y = one_hot_encoder.fit_transform(encoded_Y)
dummy_Y = pd.DataFrame(encoded_Y)
dummy_Y = pd.get_dummies(dummy_Y[0])
print(dummy_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_Y, test_size=0.2)
testModel(X_train,X_test,y_train,y_test)


