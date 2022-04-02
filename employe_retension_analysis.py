import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import seaborn as sns

data=pd.read_csv("HR_comma_sep.csv")
data

data.isnull().sum()

data["left"].value_counts()

sns.countplot(x ='left', data = data)

plt.figure(figsize=(10,10))
sns.countplot(x='Department',hue='left',data=data)

#sale deparment face high rate of employee left

plt.figure(figsize=(10,10))
sns.barplot(y ='left', x ='Department', data = data, palette ='plasma')

da=data[data['left']==1]
da.shape

data.describe()

# Approximately 24% of the employees have left the firm, which is a significant loss of talent to the firm.
# On an average the employees got a score of 0.71 which is quite good and when combined with a very low
#  standard deviation, the numbers look quite good

da.describe()
#comparing two category of employee

db=data[data['left']==0]
db.describe()

#when comparing satisfaction level people who left the company have less satisfaction level than people who remains there
#evalution point is also greater for people who stay there

data.hist(figsize=(8,8))
#checking whether data follows normal distribution

def Histogram(data,to_plot):
    for i in range(len(to_plot)):
        plt.hist(data[to_plot[i]])
        plt.axvline(data[to_plot[i]].mean(),color='r')
        plt.xlabel(to_plot[i])
        plt.show()
to_plt =['satisfaction_level','last_evaluation','average_montly_hours']
Histogram(data,to_plt) 

cor=data.corr()
cor

import seaborn as sns
sns.set(font_scale=1.0)
plt.figure(figsize=(8,8))
sns.heatmap(cor,annot=True)


#The number of projects and the average monthly hours worked by 
# an employee are positively co-related
#The level of satisfaction and 'left' variables are negatively co-related
#  which further lends evidence to our previous hypothesis.

sns.set(color_codes=True)
plot = sns.FacetGrid(data,col='left',hue='left',size=5)
plot.map(sns.kdeplot,'satisfaction_level','last_evaluation',shade=True,cmap='Blues')
plt.show()

#employee who left are categorize into three category
# Less satisfied and Under-performers
# Less satisfied and Above-average performers
# Highly satisfied and Above-average performers

dl = data[data['salary']=='low']
sns.countplot(x='Department',hue='left',data=dl)

x=adata.drop('left',axis=1)
x

y=adata["left"]
y

#from sklearn import preprocessing
#transformim=ng catagorical data into numerical data
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

l_Department=le.fit_transform(x["Department"])
l_Department

l_salary=le.fit_transform(x["salary"])
l_salary

x["Department"]=l_Department
x["salary"]=l_salary

#trail1
#spliting and training data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=98)

model1=linear_model.LogisticRegression()
model1.fit(x_train,y_train)

model1.score(x_test,y_test)

y_pred=model1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
cm

import seaborn as sns
sns.heatmap(cm,annot=True)
plt.xlabel('predictions')
plt.ylabel('Targets')
plt.figure(figsize=(50,50))





