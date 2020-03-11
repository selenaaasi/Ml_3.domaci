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


def univariate_selection(X_train,y_train):
    bestfeatures = SelectKBest(score_func=chi2, k=12)
    fit = bestfeatures.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    newScore=featureScores.nlargest(12, 'Score')
    print(featureScores.nlargest(12, 'Score'))
    return newScore

def RFE_col(X_train,y_train,n):
    classifier = LogisticRegression(random_state=0)
    rfe = RFE(classifier, n)
    fit = rfe.fit(X_train, y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    list=fit.support_
    features=[]
    for i,value in enumerate(list):
        if value == True:
            features.append(i)
    rfe_rank = fit.ranking_
    return features

def testModel(trainX,testX,trainy,testy):
    # define model
    model = tf.keras.Sequential()
    #model.add(layers.InputLayer((10,)))
    model.add(layers.Dense(10, input_dim = 10, activation = "relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(18, activation = "softmax"))

    # compile model

    model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics =['accuracy'])

    # #early stopping
    es_callback = EarlyStopping(monitor='val_loss', patience=3)

    # fit model
    history = model.fit(trainX, trainy,batch_size=64, validation_data=(testX, testy), epochs=500, verbose=0)
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


dataFrame = pd.read_csv("student-mat.csv", sep=';')
#dataFrame_2=pd.read_csv("student-por.csv", sep=';')
#dataFrame= pd.concat([dataFrame_1, dataFrame_2])
dataSet = dataFrame
X = dataSet .iloc[:, 0:32]

atributes_for_labelEncoder=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','G1','G2']
labelEncoder_X = LabelEncoder()
for a in atributes_for_labelEncoder:
    X[a] = labelEncoder_X.fit_transform(X[a])
print(X)

Y = dataSet.iloc[:, 32]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_Y = np_utils.to_categorical(encoded_Y)
print(dummy_Y[0])

score_us=univariate_selection(X, encoded_Y)
print("SCORE")
print(score_us)

X=X.iloc[:,[29,30,31,14,10,27,15,22,26,17]]
minmax_X = MinMaxScaler()
X = minmax_X.fit_transform(X)
print(X)


##kad je dummy onda je 27
#X = pd.DataFrame(X)
#print(X)
#dummy=pd.get_dummies(X,columns=['failures','reason','Walc','schoolsup','romantic','Dalc','paid'])
#print("Dummy")
#print(dummy)
#X=dummy.values


# X = pd.DataFrame(X)
# # score_us = univariate_selection(X, dummy_Y)
# # print("SCORE")
# print(score_us)
# X = X.iloc[:, [14, 31, 15, 30, 22, 17, 26, 27, 4, 1]]

X_train, X_test, y_train, y_test = train_test_split(X, dummy_Y, test_size=0.3)
testModel(X_train,X_test,y_train,y_test)



#columns_for_dummy= ['school','sex','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','G1','G2']
#columns_for_dummy_without_numeric=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
#dummy=dataFrame
#del dummy['G3']
#dummy=pd.get_dummies(dataFrame,columns=columns_for_dummy)
#X=dummy.values
#print("Dummy")
#print(X.shape)

# estimator = KerasClassifier(build_fn = testModel, epochs=200, batch_size = 32, verbose = 0)
# kfold= KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_Y, cv= kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))