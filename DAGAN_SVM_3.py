from sklearn import preprocessing
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

print("Loading DATA")
X_train_30sec_F = pd.read_csv('/home/satishk/GAN_lre/gan_csv/10sec_to_30sec_data/X_train_30sec_F.csv')

y_30sec_labels_train = pd.read_csv('/home/satishk/GAN_lre/gan_csv/10sec_to_30sec_data/y_30sec_labels_train.csv')

train_X_gen_30 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/10sec_to_30sec_data/train_X_gen.csv')

X_val_gen_30 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/10sec_to_30sec_data/X_val_gen.csv')

y_val_labels_30 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/10sec_to_30sec_data/y_val_labels.csv')

train_X_gen_10 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/3sec_to_10sec_data/train_X_gen.csv')

y_10sec_labels_train = pd.read_csv('/home/satishk/GAN_lre/gan_csv/3sec_to_10sec_data/y_10sec_labels_train.csv')

X_val_gen_10 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/3sec_to_10sec_data/X_val_gen.csv')

y_val_labels_10 = pd.read_csv('/home/satishk/GAN_lre/gan_csv/3sec_to_10sec_data/y_val_labels.csv')

train_X_gen_F = pd.concat([train_X_gen_30, train_X_gen_10, X_train_30sec_F], axis=0)

y_labels_train_F = pd.concat([y_30sec_labels_train, y_10sec_labels_train, y_30sec_labels_train], axis=0)

val_lre = pd.read_csv('/home/satishk/GAN_lre/gan_csv/dev_feat_BNF_h5_07Feb_30sec.csv')


X_val_30_orig = val_lre.drop(["language_code","uttid","segmentid1","data_source","speech_duration"],axis=1)
y_val_30_orig = val_lre["language_code"]

print("Data Loading finished!")

le = preprocessing.LabelEncoder()
le.fit(y_val_30_orig)

y_val_labels_30_orig = le.transform(y_val_30_orig)

y_val_labels_30_orig2 =y_val_labels_30_orig.reshape(933,1)

y_val_labels_30 = pd.DataFrame(y_val_labels_30.values)
y_val_labels_10 = pd.DataFrame(y_val_labels_10.values)
y_val_labels_30_orig2 = pd.DataFrame(y_val_labels_30_orig2)

y_labels_val_F = pd.concat([y_val_labels_30, y_val_labels_10, y_val_labels_30_orig2], axis=0)



X_val_F = pd.concat([X_val_gen_30, X_val_gen_10, X_val_30_orig], axis=0)

train_X_gen_F = train_X_gen_F.values
y_labels_train_F = y_labels_train_F.values
X_val_F = X_val_F.values
y_labels_val_F = y_labels_val_F.values

y_labels_train_F = y_labels_train_F.reshape(470946,)
y_labels_val_F = y_labels_val_F.reshape(2787,)

#train_X_gen_F = train_X_gen_F[0:100000]
#y_labels_train_F = y_labels_train_F[0:100000]


print("Training SVM......")

classif = SVC(C=1, kernel='linear', degree=2, gamma='auto', coef0=1, shrinking=True, random_state=0,
              probability=False, tol=1e-3, cache_size=1e4, class_weight='balanced')
classif.fit(train_X_gen_F, y_labels_train_F)

classif.classes_

acc = accuracy_score(y_labels_val_F, classif.predict(X_val_F))

print(acc)

acc



from sklearn.externals import joblib

filename = '/home/satishk/GAN_lre/gan_csv/SVM_classifier_total_linear.joblib.pkl'

_ = joblib.dump(classif, filename, compress=9)

clf2 = joblib.load(filename)
print(clf2)

acc = accuracy_score(y_labels_val_F, clf2.predict(X_val_F))
print(acc)
