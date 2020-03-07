import pandas
from sklearn.externals import joblib
import operator
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

######----default name for label is "class"--------#####

def sort_byvalue(dictx):
    dicty=sorted(dictx.items(),key=operator.itemgetter(1),reverse=True)
    return dicty


def any2list(X):
    xlist = []
    if type(X) == str: X = [X]; print("txt2vec--->string")
    if type(X) == list: xlist = X; print("txt2vec--->list")
    if type(X) == np.ndarray:
        for i in range(len(X)): xlist.append(X[i][0])
        print("txt2vec--->ndarray")
    return xlist

def tex2vec(xlist, model_name):
    Vectorizer = TfidfVectorizer(min_df=0.001, max_df=1.0, stop_words='english')
    X_train_vectors = Vectorizer.fit(xlist)
    joblib.dump(Vectorizer, 'models/' + model_name + '_vec.pkl')
    X_train_vectors = Vectorizer.transform(xlist)
    X_train_vectors = X_train_vectors.toarray()
    return X_train_vectors
    

def get_XY(url, vectorize, features):
    dataset = pandas.read_csv(url, names=features); #print(dataset); return
    if vectorize==1:
        h=list(dataset.columns.values)[0]; #print(h)
        dataset[h] = dataset[h].values.astype('U')
    array = dataset.values; #print(array[0])
    n = len(array[0]); #print("len--->", n)
    X = array[:,0:n-1]
    Y = array[:,n-1]
    return X, Y


def summarize(url, features):
    dataset = pandas.read_csv(url, names=features)
    #Summarize the Dataset
    Summary={}
    #Summary['Shape']=dataset.shape
    #Summary['Structure']=dataset.head(1)
    #Summary['Describe']=dataset.describe()
    Summary['Groups']=dataset.groupby('class').size().to_json()
    #print(Summary)
    
    #Data Visualization
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False); plt.savefig("box.jpg")
    dataset.hist(); plt.savefig("hist.jpg")
    scatter_matrix(dataset); plt.savefig("scatter.jpg")
    return {"summary":Summary, "box":"C:/services.ai/Classifier/box.jpg", "hist":"C:/services.ai/Classifier/hist.jpg", "scatter":"C:/services.ai/Classifier/scatter.jpg"}
    
 
def get_models():
    models = {}
    models['LogR'] = LogisticRegression()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['DTC'] = DecisionTreeClassifier()
    models['NBC'] = GaussianNB()
    models['SVC'] = SVC()
    return models
    

def compare(url, features, vectorize):
    X, Y = get_XY(url, vectorize, features) 
    if vectorize==1:
        xlist = any2list(X)
        X = tex2vec(xlist, "compare")
    
    #create validation set
    validation_size = 0.20
    seed = 7
    Xt, Xv, Yt, Yv = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    #Test options and evaluation metric
    scoring = 'accuracy'

    models = get_models()
    # evaluate each model in turn
    results = []
    model_names = []
    model_list = {}
    compare_list = {}
    for name, model in models.items():
        kfold = model_selection.KFold(n_splits=2, random_state=seed)
        cv_results = model_selection.cross_val_score(model, Xt, Yt, cv=kfold, scoring=scoring)
        results.append(cv_results)
        model_names.append(name)
        model_list[model]=cv_results.mean()
        cvm = round(cv_results.mean(),2); cvs = round(cv_results.std(),4)
        compare_list[name]=[" Mean: "+str(cvm), "  Std: "+str(cvs)]
        print(name, ':', cvm, cvs)
    #print(model_names)
    model_dict=sort_byvalue(model_list); #print(model_dict)
    final_model=model_dict[0][0]; #print('final_model: ',final_model)

    #Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(model_names)
    plt.savefig('comparison.jpg')
    return [compare_list, "C:/services.ai/Classifier/comparison.jpg", str(final_model)]


            
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    

def train(url, features, model_key, vectorize, model_name):
    model_dict=get_models()
    final_model=model_dict[model_key]; #print('final_model--->', final_model)
    X, Y = get_XY(url, vectorize, features)
    if vectorize==1:
        xlist = any2list(X)
        X = tex2vec(xlist, model_name)
    #print("input-->", X); print("label-->", Y)
    
    #create validation set
    tsize = 0.2
    Xt, Xv, Yt, Yv = model_selection.train_test_split(X, Y, test_size=tsize, random_state=0)
    #print(Xvalidation[0])
    ## Make predictions on validation dataset
    Yt = Yt.reshape(Yt.size, 1); #print(Yt)
    print(Xt.shape, Yt.shape);#return
    clf=final_model.fit(Xt, Yt)
    joblib.dump(clf, 'models/'+model_name+'.pkl'); print("Training Completed")
    predictions = clf.predict(Xv); #print(predictions); #print(Xv)
    score = accuracy_score(Yv, predictions); print("Score:", score)
    #report = classification_report(Yvalidation, predictions)
    #matrix = confusion_matrix(Yvalidation, predictions)
    #print('Accuracy: ', score); #print(report); print(matrix)
    
#    title = "Learning Curves - "+str(model_key)
#    # Cross validation with 2 iterations to get smoother mean test and train
#    # score curves, each time with 20% data randomly selected as a validation set.
#    cv = ShuffleSplit(n_splits=2, test_size=tsize, random_state=0)
#    plot_learning_curve(final_model, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#    plt.savefig("learning_curve.jpg")
    return [round(score,2), "C:/services.ai/Classifier/learning_curve.jpg"]


def predict(url, vectorize, model_name):
    model = joblib.load('models/' + model_name + '.pkl')

    X, Y = get_XY(url, vectorize, ['desc','temp']); #print(X)
    if vectorize==1:
        vec = joblib.load('models/' + model_name + '_vec.pkl')
        xlist = any2list(X)
        X = vec.transform(xlist)
        X = X.toarray(); #print(X)

    result = model.predict(X); print("Predictions:",result)
    #distance_by_class = model.decision_function(x); #confidence of classes
    return result
    
# compare("input/iris.csv",['a','b','c','d','e'],0)
# train("input/iris.csv",['a','b','c','d','e'],"NBC",0,"iris")
# predict("input/iris_test.csv",0,"iris")

# compare("input/hca.csv",['desc','class'],1)
# train("input/hca.csv",['desc','class'],"NBC",1,"hca")
# predict("input/hca_test.csv",1,"hca")