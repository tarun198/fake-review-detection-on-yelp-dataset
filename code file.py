# -*- coding: utf-8 -*-
"""
@author: Tarundeep Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset= pd.read_csv('metadata.csv')
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stopword_list=nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
from nltk.stem.porter import PorterStemmer
import random 
'''
#Creating Random Dataset
#FAKE
n = 80466 #number of rows in the file
s = 50000 #desired sample size
skip = random.sample(range(n),n-s)
df1 = pd.read_csv('label0.csv', skiprows=skip,names=["Reviewer ID", "Product ID", "Rating", "Label","Average R","R Deviation","Date","Review"] )

#GENUINE
n1 = 528132 #number of rows in the file
s1 = 50000 #desired sample size
skip2 = random.sample(range(n1),n1-s1)
df2 = pd.read_csv('label1.csv', skiprows=skip2,names=["Reviewer ID", "Product ID", "Rating", "Label","Average R","R Deviation","Date","Review"] )

#Concatenating the two
df=np.concatenate([df2,df1],axis=0)
df=pd.DataFrame(df)
df.to_csv('equal_values_2.csv')
'''
dataset= pd.read_csv('equal_values.csv')

#DATA PREPROCESSING
X=dataset.iloc[0:16002,[0,1,5,6,2]].values
corpus=[]
for i in range(0,16002):
        review=re.sub('[^a-zA-Z]', ' ',str(dataset['Review'][i]))
        review=review.lower()
        review=review.split()
        ps=PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopword_list)]
        review=' '.join(review) 
        corpus.append(review)

#POS TAGGING        
from nltk.tokenize import word_tokenize, sent_tokenize     
text=[]
for i in range(0,16002):
    texte = word_tokenize(dataset['Review'][i])
    text.append(texte)
    text[i]=nltk.pos_tag(text[i])

count=[]
for i in range(0,16002):
    words, tags=zip(*text[i])
    count.append(tags)
    count[i]=' '.join(count[i])

from sklearn.feature_extraction.text import CountVectorizer
tup=CountVectorizer()
X2=tup.fit_transform(count).toarray()
tup.get_feature_names()

#BIGRAMS
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(2,2),max_features=100)    
X7 = count_vect.fit_transform(corpus).toarray()
count_vect.get_feature_names()

#UNIGRAMS
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100)
X1=cv.fit_transform(corpus).toarray()
cv.get_feature_names()

#DEPENDENT VARIABLE
y1=dataset.iloc[0:16002,3].values



# positive negative
l7=[]
for i in range(0,16002):
    if(int(X[i][4])>3):
        l7.append([1])
    else:
        l7.append([0])


#CREATING INDEPENDENT VARIABLES
X1=pd.DataFrame(X1)
X7=pd.DataFrame(X7)
X2=pd.DataFrame(X2)
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
l6=[]
for i in range(0,16002):
        d,m,y=X[i][3].split("-")
        l1.append([d])
        l2.append([m])
        l3.append([y])
        l4.append([X[i][0]])
        l5.append([X[i][1]])
        l6.append([X[i][2]])        
X = pd.DataFrame(X)
dff4=np.concatenate([l1,l2,l3,l4,l5,l6,l7],axis=1)
dff4=pd.DataFrame(dff4)
dff4=np.concatenate([dff4,X1,X7,X2],axis=1)


#FINAL SET OF INDEPENDENT VARIABLES






#Logistic Regression

from sklearn.cross_validation import train_test_split
dff4=pd.DataFrame(dff4)
y=pd.Series(y1)
X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(dff4, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_L = sc.fit_transform(X_train_L)
X_test_L = sc.transform(X_test_L)

'''
#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=50)
X_train_L=lda.fit_transform(X_train_L,y_train_L)
X_test_L=lda.transform(X_test_L)
'''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_L, y_train_L)

# Predicting the Test set results
y_pred_L = classifier.predict(X_test_L)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_L, y_pred_L)

from sklearn.metrics import accuracy_score
accuracy_L = accuracy_score(y_test_L,y_pred_L)
print(accuracy_L)


#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(10):
    #print(l7[y_test_L.index[p]])
    #print(y_test_L.index[p])
    if(y_test_L[y_test_L.index[p]]==0 and l7[y_test_L.index[p]]==[1]):
        c1+=1
        if(if(y_pred_L[p]==0)
        c3=
    if(y_test_L[y_test_L.index[p]]==0 and l7[y_test_L.index[p]]==[1]):
        c2+=1
    #if(y_pred_L[p]==0 and l7[y_test_L.index[p]]==[1] and y_test_L[y_test_L.index[p]]==0):
        c3=c1+1
    if(y_pred_L[p]==0 and l7[y_test_L.index[p]]==[0] and y_test_L[y_test_L.index[p]]==0):
        c4=c1+1
print(c1)
print(c2)
print(c3)
print(c4)
print(c3/c1)
print(c4/c2)
#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_L = cross_val_score(estimator=classifier ,X= X_train_L ,y= y_train_L ,cv=5)
print(accuracies_L)
accuracies_L.max()
accuracies_L.std()

#grid
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty': ['l1'], 'C': [0.023, 0.022, 0.021, 0.008, 100]},
              {'penalty': ['l2'], 'dual': [False] ,'C': [0.01, 0.02, 0.03, 10, 100],'tol':[0.0001],'solver': ['liblinear']} ]
print('dfd')
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_L, y_train_L)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_parameters)
print(best_accuracy)






#SVM
from sklearn.cross_validation import train_test_split
X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(dff4, y1, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_S = sc.fit_transform(X_train_S)
X_test_S = sc.transform(X_test_S)



#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=50)
X_train_S=lda.fit_transform(X_train_S,y_train_S)
X_test_S=lda.transform(X_test_S)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_S, y_train_S)

# Predicting the Test set results
y_pred_S = classifier.predict(X_test_S)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_S, y_pred_S)

from sklearn.metrics import accuracy_score
accuracy_S = accuracy_score(y_test_S,y_pred_S)
print(accuracy_S)

#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(4001):
    if(y_pred_S[p]==0 and y_test_S[p]==0):
        c1+=1
        
    elif(y_pred_S[p]==1 and y_test_S[p]==1):
        c2+=1
    
    elif(y_pred_S[p]==0 and y_test_S[p]==1):
        c3+=1
    
    elif(y_pred_S[p]==1 and y_test_S[p]==0):
        c4+=1
print(c1)
print(c2)
print(c3)
print(c4)

po_fake=c1/(c4+c1)
ne_fake=c2/(c3+c2)
print(po_fake)
print(ne_fake)



#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_S = cross_val_score(estimator=classifier ,X= X_train_S ,y= y_train_S ,cv=5)
print(accuracies_S)
accuracies_S.mean()
accuracies_S.std()


#naive_bayes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dff4, y1, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

'''#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=150)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)
'''


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("asd",accuracy)

#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(4001):
    if(y_pred[p]==0 and y_test[p]==0):
        c1+=1
        
    elif(y_pred[p]==1 and y_test[p]==1):
        c2+=1
    
    elif(y_pred[p]==0 and y_test[p]==1):
        c3+=1
    
    elif(y_pred[p]==1 and y_test[p]==0):
        c4+=1
print(c1)
print(c2)
print(c3)
print(c4)

po_fake=c1/(c4+c1)
ne_fake=c2/(c3+c2)
print(po_fake)
print(ne_fake)



#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier ,X= X_train ,y= y_train ,cv=5)
print(accuracies)
accuracies.mean()
accuracies.std()


#Decision tree
from sklearn.cross_validation import train_test_split
X_train_D, X_test_D, y_train_D, y_test_D = train_test_split(dff4, y1, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_D = sc.fit_transform(X_train_D)
X_test_D = sc.transform(X_test_D)
'''
#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=150)
X_train_D=lda.fit_transform(X_train_D,y_train_D)
X_test_D=lda.transform(X_test_D)
'''


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train_D, y_train_D)
y_pred_D = classifier.predict(X_test_D)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_D = confusion_matrix(y_test_D, y_pred_D)
from sklearn.metrics import accuracy_score
accuracy_D = accuracy_score(y_test_D,y_pred_D)
print("asd",accuracy_D)

#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(4001):
    if(y_pred_D[p]==0 and y_test_D[p]==0):
        c1+=1
        
    elif(y_pred_D[p]==1 and y_test_D[p]==1):
        c2+=1
    
    elif(y_pred_D[p]==0 and y_test_D[p]==1):
        c3+=1
    
    elif(y_pred_D[p]==1 and y_test_D[p]==0):
        c4+=1
print(c1)
print(c2)
print(c3)
print(c4)

po_fake=c1/(c4+c1)
ne_fake=c2/(c3+c2)
print(po_fake)
print(ne_fake)



#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_D = cross_val_score(estimator=classifier ,X= X_train ,y= y_train ,cv=5)
print(accuracies_D)
accuracies_D.mean()
accuracies_D.std()



#Random Forest
from sklearn.cross_validation import train_test_split
X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(dff4, y1, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_R = sc.fit_transform(X_train_R)
X_test_R = sc.transform(X_test_R)

#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=150)
X_train_R=lda.fit_transform(X_train_R,y_train_R)
X_test_R=lda.transform(X_test_R)



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,criterion = 'entropy', random_state = 0)
classifier.fit(X_train_R, y_train_R)

# Predicting the Test set results
y_pred_R = classifier.predict(X_test_R)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_R, y_pred_R)
accuracy_R = accuracy_score(y_test_R,y_pred_R)
print(accuracy_R)
#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(4001):
    if(y_pred_R[p]==0 and y_test_R[p]==0):
        c1+=1
        
    elif(y_pred_R[p]==1 and y_test_R[p]==1):
        c2+=1
    
    elif(y_pred_R[p]==0 and y_test_R[p]==1):
        c3+=1
    
    elif(y_pred_R[p]==1 and y_test_R[p]==0):
        c4+=1
print(c1)
print(c2)
print(c3)
print(c4)

po_fake=c1/(c4+c1)
ne_fake=c2/(c3+c2)
print(po_fake)
print(ne_fake)




#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_R = cross_val_score(estimator=classifier ,X= X_train_R ,y= y_train_R ,cv=5)
print(accuracies_R)
accuracies_R.mean()
accuracies_R.std()





#Kernel SVM
from sklearn.cross_validation import train_test_split
X_train_K, X_test_K, y_train_K, y_test_K = train_test_split(dff4, y1, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train_K)
X_test = sc.transform(X_test_K)

#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=150)
X_train_K=lda.fit_transform(X_train_K,y_train_K)
X_test_K=lda.transform(X_test_K)



# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train_K, y_train_K)

# Predicting the Test set results
y_pred_K = classifier.predict(X_test_K)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_K, y_pred_K)
accuracy_K = accuracy_score(y_test_K,y_pred_K)
print(accuracy_K)


#positive negative fake accuracy calculation
c1=c2=c3=c4=0
for p in range(3201):
    if(y_pred_K[p]==0 and y_test_K[p]==0):
        c1+=1
        
    elif(y_pred_K[p]==1 and y_test_K[p]==1):
        c2+=1
    
    elif(y_pred_K[p]==0 and y_test_K[p]==1):
        c3+=1
    
    elif(y_pred_K[p]==1 and y_test_K[p]==0):
        c4+=1
print(c1)
print(c2)
print(c3)
print(c4)

po_fake=c1/(c4+c1)
ne_fake=c2/(c3+c2)
print(po_fake)
print(ne_fake)



#applying k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_K = cross_val_score(estimator=classifier ,X= X_train_K ,y= y_train_K ,cv=5)
print(accuracies_K)
accuracies_K.mean()
accuracies_K.std()
