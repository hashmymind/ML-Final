from sklearn.naive_bayes import GaussianNB
from data import *

def get_classifier(train_X,train_y):
    gnb = GaussianNB()
    return gnb.fit(train_X, train_y)
    
    
if __name__ == '__main__':
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    
    
    
    (train_X, train_y), (test_X, test_y) = get_sst()
    corpus = train_X + test_X
    vectorizer = TfidfVectorizer(min_df = 5).fit(corpus)
    train_X = vectorizer.transform(train_X).toarray()
    test_X = vectorizer.transform(test_X).toarray()
    model = get_classifier(train_X,train_y)
    y_pred = model.predict(test_X)
    target_names = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    print(classification_report(test_y, y_pred, target_names=target_names))
    print(confusion_matrix(test_y, y_pred))
    print(accuracy_score(test_y, y_pred))