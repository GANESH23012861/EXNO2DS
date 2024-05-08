# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![i1](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/5ded3eed-9580-4d52-b4d4-46bd0800e0fb)



```
dt.info()
```
![i2](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/93d39005-1723-4dd1-bd9e-d982593fa289)



```
dt.shape
```
![i3](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/9203b783-f251-4e12-96c7-55b9baea2bae)



```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![i4](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/0e27de34-9652-477f-828d-e81cc82092c1)



```
dt.nunique()
```
![i5](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/b6e4e3ae-ca9a-4d0d-a8d2-961b55fee46d)



```
dt["Survived"].value_counts()
```
![i6](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/0b3d5011-c153-4fab-b113-ef314ba17b49)



```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![i7](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/9ce96e1c-e629-4f4d-9ae6-8f95d4ef275f)



```
sns.countplot(data=dt,x="Survived")
```
![i8](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/5b0a579e-e801-4513-b7f9-2d0be7c09b53)



```
dt
```
![i9](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/367bf50a-a8e8-41a4-bea9-c76c0545f677)




```
dt.Pclass.unique()
```
![i10](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/c6fd8302-cbf6-4298-97a5-5495e64c796f)



```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![i11](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/d6b25f8b-6cf2-4fed-aa40-deba76670100)



```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![i12](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/33fb73dc-9f31-4a04-a0e6-f2be064333e1)



```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![i13](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/97c65d98-3d5b-4c8f-b15b-133eaa44ff36)



```
dt.boxplot(column="Age",by="Survived")
```
![i14](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/dfc871b2-3bb4-48fa-9c54-1bce095aaefa)



```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![i15](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/cb75f5dc-d861-4f46-bf25-0968beebcec3)



```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![i16](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/fd3e5f25-ef67-4b94-b5d0-d70aa96c6899)



```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![i17](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/d6682085-efab-4ffd-90e1-671d50df4117)



```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![i18](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/84bb21a0-ff01-46fb-8b07-dd7c6177db8e)



```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![i19](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/609e683c-fa96-4522-93fd-4aefede06e48)



```
sns.pairplot(dt)
```
![i20](https://github.com/silambarasan2004/EXNO2DS/assets/119559917/1ac874ab-6bd4-4a5f-bebe-a5f403832beb)


# RESULT
       Thus, the Exploratory Data Analysis on the given data set was performed successfully. 








