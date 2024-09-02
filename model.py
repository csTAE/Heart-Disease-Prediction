import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report
import pickle


df = pd.read_csv('heart.csv')

X = df.drop(columns = ["target"])
y = df["target"]

smote=SMOTE(sampling_strategy='minority')
X,y=smote.fit_resample(X,y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state = 0)


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model= RandomForestClassifier(n_estimators= 10, criterion="entropy") 
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred))

pickle.dump(model, open('model1.pkl', 'wb'))
model1=pickle.load(open('model1.pkl','rb'))
                   

