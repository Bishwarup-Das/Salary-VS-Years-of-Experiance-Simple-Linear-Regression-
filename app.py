import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

Data_Frame= pd.read_csv("Salary_Data.csv")

X= Data_Frame.iloc[:,:-1].values

y= Data_Frame.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)

scaler= StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test= scaler.transform(X_test)

regression= LinearRegression()

regression.fit(X_train,y_train)

print("What should be your salary based on experience?")
ex=int(input("Ente the Experience: "))

new_exp=scaler.transform([[ex]])
out=regression.predict(new_exp)

print("The Salary should be: ", int(out), "Rupees")