# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
df1=pd.read_csv("C:\\Users\\admin\\Downloads\\bmi.csv")
df1
```
<img width="701" height="573" alt="image" src="https://github.com/user-attachments/assets/773134d5-e271-4920-8bc1-f96872afec3b" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler
df2=df1.copy()
enc=StandardScaler()
df2[['new_height','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df2
```
<img width="1082" height="620" alt="image" src="https://github.com/user-attachments/assets/f8fc940a-fa11-45ec-8501-7a355b53d409" />

```
df3=df1.copy()
enc=MinMaxScaler()
df3[['new_height','new_weight']]=enc.fit_transform(df3[['Height','Weight']])
df3
```
<img width="908" height="569" alt="image" src="https://github.com/user-attachments/assets/3c4de077-b7b7-4ce6-affd-97eb0d4e917c" />

```
df4=df1.copy()
enc=MaxAbsScaler()
df4[['new_height','new_weight']]=enc.fit_transform(df3[['Height','Weight']])
df4
```
<img width="1077" height="555" alt="image" src="https://github.com/user-attachments/assets/3144c433-66ec-4697-aa8a-a5695df46423" />

```
df5=df1.copy()
enc=Normalizer()
df5[['new_height','new_weight']]=enc.fit_transform(df5[['Height','Weight']])
df5
```
<img width="868" height="567" alt="image" src="https://github.com/user-attachments/assets/d780a25f-6032-4b80-8235-6ef4a4b6899a" />

```
df6=df1.copy()
enc=RobustScaler()
df6[['new_height','new_weight']]=enc.fit_transform(df6[['Height','Weight']])
df6
```
<img width="959" height="565" alt="image" src="https://github.com/user-attachments/assets/9bc617fd-60d2-432b-b783-3dc8b09ba156" />

```
df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
df
```
<img width="1387" height="754" alt="image" src="https://github.com/user-attachments/assets/0753e533-94ea-48ef-92b1-5f7b85c6e70c" />

```
from sklearn.preprocessing import LabelEncoder
df_encoded=df.copy()
le=LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])
    
x=df_encoded.drop("SalStat",axis=1)
y=df_encoded["SalStat"]
```
```
x
```
<img width="1236" height="506" alt="image" src="https://github.com/user-attachments/assets/2f158d55-7980-4c94-939c-43a263906608" />

```
from sklearn.feature_selection import SelectKBest, chi2

chi2_selector=SelectKBest(chi2,k=5)
chi2_selector.fit(x,y)

selected_features_chi2=x.columns[chi2_selector.get_support()]
print("Selected features (Chi-Square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```
<img width="1199" height="553" alt="image" src="https://github.com/user-attachments/assets/520302ef-96ae-44e4-b364-caffec3307c0" />

```
from sklearn.feature_selection import f_classif

anova_selector=SelectKBest(f_classif,k=5)
anova_selector.fit(x,y)

selected_features_anova=x.columns[anova_selector.get_support()]
print("Selected features (ANOVA F-test):",list(selected_features_anova))

mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```
<img width="1118" height="537" alt="image" src="https://github.com/user-attachments/assets/e2b568ac-11f9-40d9-8deb-357269ca8b45" />

```
from sklearn.feature_selection import mutual_info_classif
mi_selector=SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]
print("Selected features (Mutual Info):",list(selected_features_mi))
mi_scores=pd.Series(mi_selector.scores_,index=x.columns)
print("\nMutual Information Scores:\n",mi_scores.sort_values(ascending=False))
```
<img width="1131" height="551" alt="image" src="https://github.com/user-attachments/assets/c1532725-d501-4cfe-be5a-01aca04ade99" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE 

model = LogisticRegression(max_iter=100)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe=x.columns[rfe.support_]
print("Selected features (RFE):", list(selected_features_rfe))
```
<img width="1397" height="256" alt="image" src="https://github.com/user-attachments/assets/5352e3f5-76ae-4a92-b8cc-0e5a431cdbff" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(max_iter=100)
rfe = SequentialFeatureSelector(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe=x.columns[rfe.support_]
print("Selected features (SF):", list(selected_features_rfe))
```
<img width="1375" height="641" alt="image" src="https://github.com/user-attachments/assets/410ded35-5227-46af-8f31-98273d017c86" />

```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x,y)
importances=pd.Series(rf.feature_importances_,index=x.columns)
selected_features_rf=importances.sort_values(ascending=False)
print(importances)
print("Selected features (RandomForestClassifier):",list(selected_features_rf))
```
<img width="1338" height="523" alt="image" src="https://github.com/user-attachments/assets/9db22a1d-cb19-4a47-8ffd-d9df1a8a1a20" />

```
from sklearn.linear_model import LassoCV
import numpy as np
lasso=LassoCV(cv=5).fit(x,y)
importance=np.abs(lasso.coef_)
selected_features_lasso=x.columns[importance>0]
print("Selected features (lasso):",list(selected_features_lasso))
```
<img width="985" height="190" alt="image" src="https://github.com/user-attachments/assets/71981c25-7115-4fb8-8f8e-49320d3e419f" />

```
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
le=LabelEncoder()
df_encoded=df.copy()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])

x=df_encoded.drop("SalStat",axis=1)
y=df_encoded["SalStat"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
```
<img width="790" height="355" alt="image" src="https://github.com/user-attachments/assets/f0e607b6-4fe5-4725-bd30-1a90948a814d" />


# RESULT:
<img width="790" height="355" alt="image" src="https://github.com/user-attachments/assets/50e8afb7-9639-423a-b579-4df64e37ac01" />
