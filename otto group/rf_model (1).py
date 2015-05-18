import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.svm import SVC as SVM_Classifier


BASE_DIR = "/Users/kien/Documents/otto_group/"
INPUT_TRAIN = "Train.csv"; INPUT_TEST = "Test.csv"


class Otto_Group():
  def __init__(self):
    self.k_folds = 10; 
    self.seed = 95127            # 90007; 92126; 94085; 93103
    self.train = pd.DataFrame()  # train data
    self.test = []               # test data
    self.y_pred = []             # prediction for train set
    self.y_true = []             # train labels
    self.n_samples = 0           # number of training samples
    self.num_class = 0           # number of classes
    self.ids = []                # train index
    self.idx = []                #  test index
    
  def logloss(self, y_true, y_pred):
    '''
    Parameters:
    ----------
      y_true: {array-like} shape {n_samples}
      y_pred: {array-like} shape {n_samples, n_classes}

    Return: logloss
    ------
    '''
    epsilon=1e-15
    
    # get probabilities
    y = [np.maximum(epsilon, np.minimum(1-epsilon, y_pred[i, j])) 
                               for (i, j) in enumerate(self.y_true)]
    
    logloss = - np.mean(np.log(y))
    return logloss

  
  def load_data(self):
    # preprocessing train data
    df = pd.read_csv(BASE_DIR + INPUT_TRAIN)
    X = df.values.copy()
    np.random.shuffle(X)
    self.ids, X, labels = X[:, 0], X[:, 1:-1].astype(np.float32), X[:, -1]
    self.encoder = LabelEncoder()
    self.y_true = self.encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    self.train = scaler.fit_transform(X)
    
    # preprocessing test data
    df = pd.read_csv(BASE_DIR + INPUT_TEST)
    X = df.values.copy()
    X, self.idx = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    self.test = scaler.transform(X)
    
    self.num_class = len(self.encoder.classes_)
    self.n_samples = len(self.y_true)
    self.y_pred = np.zeros( (self.n_samples, self.num_class) )
    return None
    
  def prediction(self, model, **kwargs):
    clf = model(**kwargs)
    
    # 10-fold cross validation for single outcome
    cv = KFold(self.n_samples, n_folds=self.k_folds, shuffle=True, random_state=self.seed)
    preds_k = np.zeros( (len(self.test), self.num_class, self.k_folds) )
    
    for k, (tr_idx, cv_idx) in enumerate(cv):
      x_train, x_vali = self.train[tr_idx], self.train[cv_idx]
      y_train = self.y_true[tr_idx]
      
      clf.fit( x_train, y_train )
      self.y_pred[cv_idx, :] = clf.predict_proba(x_vali)
      
      # predicting on test set
      preds_k[:,:,k] = clf.predict_proba( self.test )
      print 'Complete {}th fold'.format(k+1)
      
    self.preds = preds_k.mean(2)

    logloss = self.logloss(self.y_true, self.y_pred)
    print 'Accuracy: {}.'.format(logloss)
    return None

  def save_prediction(self):
    # saving cross validation score
    cv = pd.DataFrame(self.y_pred, index=self.ids, columns = self.encoder.classes_)
    cv['target'] = self.y_true
    cv.sort_index(axis=0, ascending=True, inplace=True)
    cv.to_csv(BASE_DIR+'rf_model/pred_cv.csv', index=True, index_label='id')

    # saving prediction score
    rs = pd.DataFrame(self.preds, index=self.idx, columns = self.encoder.classes_)
    rs = rs.div(rs.sum(axis=1), axis=0)
    rs.to_csv(BASE_DIR+'rf_model/resl_randomforest_avg.csv', index=True, index_label='id')
    return None



obj = Otto_Group()
obj.load_data()
obj.prediction(RF, n_estimators=1500, n_jobs=-1)
obj.save_prediction()


