from sklearn.datasets import load_svmlight_file
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def Data_Generation ():
    X, Y= load_svmlight_file("mushrooms.txt")
    x_train, x_test, y_train, y_test \
        = train_test_split(X, Y, test_size=0.3)
    print("The size of the data set is:", len(Y))
    print("Training data size:", len(y_train))
    print("Test data size:", len(y_test))
    return x_train, y_train, x_test, y_test

def Decision_Tree():
    x_train, y_train, x_test, y_test = Data_Generation() #Data Generation
    clf = tree.DecisionTreeClassifier(max_depth=4) #Model Selection
    clf = clf.fit(x_train,y_train) #Model Fit with training data
    y_pred = clf.predict(x_test) #Test data prediction
    Accuracy = metrics.accuracy_score(y_test, y_pred) #error Calculation
    print("Accuracy value for DT:", Accuracy*100, "%")
    Error = (1 - Accuracy)
    print("Error value for DT:", Error*100,"%")
    tree.plot_tree(clf) #Tree plots print("...") for values in terminal
    plt.show()
    return Error

Decision_Tree()

#from sklearn.tree.export import export_text
#print (export_text(clf))  #Plot the decision tree in terminal into the Decision_Tree(N)
