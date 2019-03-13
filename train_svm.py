from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pandas as pd


def load_data_train():
    train_data = []
    train_label = []
    df_train = pd.read_csv("data/train.csv")
    for index, row in df_train.iterrows():
        train_data.append(row['question_text'])
        train_label.append(row['target'])
    return train_data, train_label


def train():
    train_data, train_labels = load_data_train()
    vectorizer = TfidfVectorizer(ngram_range=(1, 6))
    vectorizer.fit(train_data)
    train_data = vectorizer.transform(train_data)
    clf = LinearSVC(verbose=True)
    clf.fit(train_data, train_labels)
    joblib.dump(vectorizer, "model/vectorzier.intend.pkl")
    joblib.dump(clf, "model/classify.pkl")


def svc_param_selection(kfolds):
    cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    train_data, train_labels = load_data_train()
    vectorizer = joblib.load("model/vectorzier_quora.pkl")
    train_data = vectorizer.transform(train_data)
    param_grid = {'C': cs, 'gamma': gammas}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=kfolds, verbose=True)
    print("---------------START---------------")
    clf.fit(train_data, train_labels)
    joblib.dump(clf, "model/grid_quora.pkl")


def test():
    vectorizer = joblib.load("model/vectorzier_quora.pkl")
    clf = joblib.load("model/grid_quora.pkl")
    with open('result/result.csv', 'w') as f1:
        f1.write("qid,prediction\n")
        df_test = pd.read_csv('data/test.csv')
        for index, row in df_test.iterrows():
            vector = vectorizer.transform([row['question_text']])
            label = clf.predict(vector)
            f1.write(row['qid']+","+str(label[0]))
            f1.write("\n")


print("-----------------Load training data----------------")
load_data_train()
print("-------------------Training--------------------")
svc_param_selection(5)
print("-------------------Testing---------------------")
test()