
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

dataframe = mat.copy()
dummy = pd.get_dummies(dataframe, columns=['G3'])
print("Dummy G3")
print(dummy.shape)
labels = dummy[dummy.columns[32:50]]
X = dummy[dummy.columns[0:30]]
# labels = dataframe.pop('G3')
print("LABELS")
print(labels)
print("DATAFRAME")
print(dataframe)

feature_columns = []

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




X_train,X_test, y_train,y_test,= train_test_split(X,labels, test_size=0.2)
input_func=tf.compat.v1.estimator.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model=tf.estimator.LinearClassifier(feature_columns=feature_columns,n_classes=20)
model.train(input_fn=input_func,steps=1000)

pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
predictions=list(model.predict(input_fn=pred_input_func))

final_preds=[]
for pred in predictions:
    final_preds.append(pred['class_ids'][0])
print(final_preds[:10])

from sklearn.metrics import classification_report

print(classification_report(y_test,final_preds)


