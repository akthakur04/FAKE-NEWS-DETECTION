import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
#get dataset
df = pd.read_csv("news.csv")
#get dataset info
print(df.shape)
print(df.head(20))
labels=df.label
print(labels.head(20))
print(df.info())

#train dataset
x=df['text']
y=labels
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)


#tfidfvectorization
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

#fit and transform tfidf
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

passiveclassifier=PassiveAggressiveClassifier(max_iter=50)
passiveclassifier.fit(tfidf_train,y_train)

y_pred=passiveclassifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)

print(f'Accuracy: {round(score*100,2)}%')
# Build confusion matrix
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))
