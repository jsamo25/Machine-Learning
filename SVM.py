import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import random
from matplotlib.colors import Normalize

def Data_Generation ():
    X, Y= load_svmlight_file("fourclass.txt")
    print("the size of the data set is:", len(Y))
    return X, Y

def SVM(gamma, C):
    X, Y = Data_Generation()
    clf = svm.SVC(gamma= gamma , kernel="rbf", C= C)
    clf.fit(X, Y)
    y_pred = clf.predict(X)
    Accuracy = metrics.accuracy_score(Y,y_pred)
    print("The accuracy value for iteration is:", Accuracy)
    score =  cross_val_score(clf, X, Y, cv=3)
    print("The Cross Validation Score for this iteration is:", score)
    return X, Y

def SVM_Optimizer():
    Gamma = np.geomspace(0.0001, 100, 20)
    C_vector = np.geomspace(0.01, 1000, 20)
    gamma = random.choice(Gamma)
    C = random.choice(C_vector)
    print("the Initial Values for C and Gamma are:",
          gamma, C, "respectively")
    X, Y = SVM(gamma, C)
    param_grid = dict(gamma = Gamma, C = C_vector) #log scale
    Cross_val = StratifiedShuffleSplit(n_splits = 5,
                                       test_size = 0.3,
                                       random_state = 20)
    grid = GridSearchCV(SVC(), param_grid = param_grid,
                        cv = Cross_val)
    grid.fit(X,Y)
    print (grid)
    print ("the best parameters", (grid.best_params_, grid.best_score_))
    print ("Updating the Gamma & C Hyperparameters for this dataset...")
    SVM(grid.best_params_["gamma"], grid.best_params_["C"])
    return Gamma, C_vector, grid

class Normalize_graph(Normalize):
    def __init__(self, vmin=None, vmax = None, midpoint=None, clip = False):
        self.midpoint = midpoint
        Normalize.__init__(self,vmin,vmax,clip)
    def __call__(self, value, clip = None):
        x, y = [self.vmin,self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x,y))

def Heat_Map(gamma, C, grid):
    scores = grid.cv_results_["mean_test_score"].reshape(20, 20)
    mean = scores.mean()
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=Normalize_graph(vmin=0.2, midpoint=mean))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(20), gamma, rotation=45)
    plt.yticks(np.arange(20), C)
    plt.title('Validation accuracy')
    plt.show()

##########  INIT  #############
gamma, C, grid = SVM_Optimizer()
Heat_Map(gamma, C, grid)