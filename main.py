
# example of tf.keras python idiom
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
mat = pd.read_csv("student-mat.csv",sep=';')
por = pd.read_csv("student-por.csv",sep=';')
print(mat.head())
#merged = pd.merge(left=mat,right=por, on=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','nursery','internet','traveltime','romantic','famrel','freetime','goout','Dalc','Walc','health'])
#print(merged)

#split dataset in train,val i test set
dataframe = mat.copy()
dummy = pd.get_dummies(dataframe, columns=['G3'])
print("Dummy G3")
print(dummy.shape)
labels = dummy[dummy.columns[32:50]]
dataframe = dummy[dummy.columns[0:30]]
# labels = dataframe.pop('G3')
print("LABELS")
print(labels)
print("DATAFRAME")
print(dataframe)




train, test = train_test_split(mat, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#create an input pipeline using tf.data
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  dataframeG3 = dataframe.copy()
  labels2 = dataframeG3.pop('G3')
  print("LABELS2")
  print(labels2)
  dummy = pd.get_dummies(dataframe, columns=['G3'])
  print("Dummy G3")
  print(dummy.shape)
  dfLab = dummy.rename(columns=dummy.iloc[0]).drop(dummy.index[0])
  labels=dfLab[dfLab.columns[32:50]]
  print("DFLab"
        )
  print(dfLab)
  print(dfLab.columns)
  print(dfLab.shape)
  dataframe=dummy[dummy.columns[0:30]]
  print("LABELS")
  print(labels)
  print(labels.shape)
  print("DATAFRAME")
  print(dataframe)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

feature_columns = []

# numeric cols
# izbrisao sam g1 i g2 odavde
for header in ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences' ]:
  feature_columns.append(feature_column.numeric_column(header))

school = feature_column.categorical_column_with_vocabulary_list(
      'school', ['GP', 'MS'])
school_one_hot = feature_column.indicator_column(school)
feature_columns.append(school_one_hot)

sex = feature_column.categorical_column_with_vocabulary_list(
      'sex', ['F', 'M'])
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

address =feature_column.categorical_column_with_vocabulary_list(
      'address', ['U', 'R'])
address_one_hot = feature_column.indicator_column(address)
feature_columns.append(address_one_hot)

famsize=feature_column.categorical_column_with_vocabulary_list(
      'famsize', ['LE3', 'GT3'])
famsize_one_hot = feature_column.indicator_column(famsize)
feature_columns.append(famsize_one_hot)

Pstatus=feature_column.categorical_column_with_vocabulary_list(
      'Pstatus', ['T', 'A'])
Pstatus_one_hot = feature_column.indicator_column(Pstatus)
feature_columns.append(Pstatus_one_hot)


Mjob=feature_column.categorical_column_with_vocabulary_list(
      'Mjob', ['teacher', 'health', 'services', 'home','other'])
Mjob_one_hot = feature_column.indicator_column(Mjob)
feature_columns.append(Mjob_one_hot)

Fjob=feature_column.categorical_column_with_vocabulary_list(
      'Fjob', ['teacher', 'health', 'services', 'home','other'])
Fjob_one_hot = feature_column.indicator_column(Fjob)
feature_columns.append(Fjob_one_hot)

reason= feature_column.categorical_column_with_vocabulary_list(
      'reason', [ 'home', 'reputation', 'course', 'other'])
reason_one_hot = feature_column.indicator_column(reason)
feature_columns.append(reason_one_hot)

guardian=feature_column.categorical_column_with_vocabulary_list(
      'guardian', [ 'mother', 'father', 'other'])
guardian_one_hot = feature_column.indicator_column(guardian)
feature_columns.append(guardian_one_hot)

schoolsup=feature_column.categorical_column_with_vocabulary_list(
      'schoolsup', [ 'yes', 'no'])
schoolsup_one_hot = feature_column.indicator_column(schoolsup)
feature_columns.append(schoolsup_one_hot)

famsup=feature_column.categorical_column_with_vocabulary_list(
      'famsup', [ 'yes', 'no'])
famsup_one_hot = feature_column.indicator_column(famsup)
feature_columns.append(famsup_one_hot)

paid=feature_column.categorical_column_with_vocabulary_list(
      'paid', [ 'yes', 'no'])
paid_one_hot = feature_column.indicator_column(paid)
feature_columns.append(paid_one_hot)

activities=feature_column.categorical_column_with_vocabulary_list(
      'activities', [ 'yes', 'no'])
activities_one_hot = feature_column.indicator_column(activities)
feature_columns.append(activities_one_hot)

nursery=feature_column.categorical_column_with_vocabulary_list(
      'nursery', [ 'yes', 'no'])
nursery_one_hot = feature_column.indicator_column(nursery)
feature_columns.append(nursery_one_hot)

higher=feature_column.categorical_column_with_vocabulary_list(
      'higher', [ 'yes', 'no'])
higher_one_hot = feature_column.indicator_column(higher)
feature_columns.append(higher_one_hot)

internet=feature_column.categorical_column_with_vocabulary_list(
      'internet', [ 'yes', 'no'])
internet_one_hot = feature_column.indicator_column(internet)
feature_columns.append(internet_one_hot)

romantic=feature_column.categorical_column_with_vocabulary_list(
      'romantic', [ 'yes', 'no'])
romantic_one_hot = feature_column.indicator_column(romantic)
feature_columns.append(romantic_one_hot)

batch_size = 8
train_ds = df_to_dataset(train, batch_size=batch_size)
print(train_ds)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(32,input_dim = 31, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(17, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=200)
#
# #Aleksin deo
# def testModel():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(8, input_dim = 31, activation = "relu"))
#     model.add(layers.Dense(8, activation="relu"))
#     model.add(layers.Dense(18, activation = "softmax"))
#     model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
#     return model
# dataFrame = pd.read_csv("student-mat.csv", sep=';')
# dataSet = dataFrame.values
# X = dataSet [:, 0:31]
# indexes_for_labelEncoder=[0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22]
# labelEncoder_X = LabelEncoder()
# for i in indexes_for_labelEncoder:
#   X[:,i] = labelEncoder_X.fit_transform(X[:,i])
# minmax_X = MinMaxScaler()
# X = minmax_X.fit_transform(X)
# print("XXXXXXX")
# print (X.shape)
# print(X)
# #columns_for_dummy= ['school','sex','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','G1','G2']
# #columns_for_dummy_without_numeric=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
# #dummy=dataFrame
# #del dummy['G3']
# #dummy=pd.get_dummies(dataFrame,columns=columns_for_dummy)
# #X=dummy.values
# #print("Dummy")
# #print(X.shape)
#
# Y = dataSet[: , 32]
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# dummy_Y = np_utils.to_categorical(encoded_Y)
# print(X)
# print(dummy_Y[0])
#
# estimator = KerasClassifier(build_fn = testModel, epochs=200, batch_size = 8, verbose = 0)
# kfold= KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_Y, cv= kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
