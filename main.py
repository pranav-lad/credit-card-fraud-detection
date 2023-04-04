import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import streamlit as st
%matplotlib inline
df=pd.read_csv('creditcard.csv')
df.head()
df.describe()
sns.jointplot(x='Amount', y='Class', data=df)
from sklearn.preprocessing import RobustScaler

rbs = RobustScaler()

df_small = df[['Time','Amount']]
df_small = pd.DataFrame(rbs.fit_transform(df_small))

df_small.columns = ['scaled_time','scaled_amount']
df = pd.concat([df,df_small],axis=1)

df.drop(['Time','Amount'],axis=1,inplace=True)

df.head()
df['Class'].value_counts()
sns.countplot(df['Class'])
non_fraud = df[df['Class']==0]
fraud = df[df['Class']==1]

non_fraud = non_fraud.sample(frac=1)

non_fraud = non_fraud[:492]

new_df = pd.concat([non_fraud,fraud])
new_df = new_df.sample(frac=1)

new_df['Class'].value_counts()
new_df = new_df.reset_index(drop=True)
sns.countplot(new_df['Class'])
X = new_df.drop('Class',axis=1)
y = new_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)

st.write(classification_report(y_test,pred))
st.write('\n\n')
st.write(confusion_matrix(y_test,pred))
st.write('\n')
st.write('accuracy is --> ',round(accuracy_score(y_test,pred)*100,2))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

pred = dt.predict(X_test)

st.write(classification_report(y_test,pred))
st.write('\n\n')
st.write(confusion_matrix(y_test,pred))
st.write('\n')
st.write('accuracy is --> ',round(accuracy_score(y_test,pred)*100,2))
from sklearn.ensemble import RandomForestClassifier,IsolationForest

rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)

pred=rf.predict(X_test)

st.write(classification_report(y_test,pred))
st.write('\n\n')
st.write(confusion_matrix(y_test,pred))
st.write('\n')
st.write('accuracy is --> ',round(accuracy_score(y_test,pred)*100,2))
