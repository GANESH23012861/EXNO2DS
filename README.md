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
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![311415580-27aae021-a93c-4d62-8700-79b1beaba84e](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/4f626cab-2e6b-42c2-ac3e-a228053b37da)

```python
dt.info()
```
![311415718-23c9a813-e29e-458e-b872-c0e96d2b75d4](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/06234124-56ee-4f63-a415-a6ff4d12c1b0)

```python
dt.shape
```
![311415809-9b3f8940-f34a-4efe-857c-5c3e286a59f2](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/0a66836a-f04a-4bae-8c75-b788d085ae39)

```python
dt.describe()
```
![311417476-603a6476-0b6e-4939-be1d-020a5411eb5a](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/97ca288c-a3fb-4823-b949-b1b1c42ee839)

```python
dt.nunique()
```
![311417491-e5e1049e-0421-4ed2-9d7c-15e0a7f0317b](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/39aa2fb7-ddba-47bd-8afd-193fe2332a3b)

```python
dt["Survived"].value_counts()
```
![311415948-19899d18-deb2-459d-8d6f-9610872d098f](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/d4c84ffc-73b5-41d4-89b6-d6837c5ef0e6)

```python
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![311416106-453f521f-e016-450a-823e-853513864c8a](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/1e2e7907-8d31-434d-a6c2-c37df5e12495)

```python
sns.countplot(data=dt,x="Survived")
```
![311416143-b38c5a88-9b85-4e68-a1fd-f25fb35da7d5](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/87a62661-6655-410d-b44a-ed0943326080)

```python
dt
```
![311416203-75851857-515c-45a4-9ed0-f97bc501ac02](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/714ef9f7-0a11-4458-a689-54cb94eaea8e)


```python
dt.Pclass.unique()
```
![311416245-4868a17d-a4de-45c3-86e4-c769762d77ca](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/31f88729-3e19-47fa-8fc6-9bb76e50ae8a)

```python
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![311416315-822c7791-fb41-44bb-8c23-8b8d1a3d811c](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/01167250-0a18-4a76-97d0-997ecc720a7f)

```python
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5, aspect=.7)

```
![311416381-fdadc299-3453-47ae-86cd-8bd9e77120a1](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/ec2fca03-4b2d-40e3-a481-d024be37d848)

```python
sns.catplot(x='Survived',hue="Gender",data=dt,kind="count")
```
![311416409-7fb380ba-e219-4d79-aa78-c9591ddcb8b6](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/71985d8c-5971-48c3-8b40-dd0ac3638ea8)

```python
dt.boxplot(column="Age",by="Survived")
```
![311416457-c84bad98-5cd5-4faf-9d15-85d0645205d9](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/f70f0ab6-cf4d-4ef6-a1a1-d6dabd45cba8)

```python
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![311416485-089a4887-89a2-47b9-9f11-ef75e3016301](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/b6fdbb75-f867-4da0-9aba-27586c1c17b8)

```python
sns.jointplot(x="Age",y="Fare",data=dt)
```
![311416523-3b43d8e9-fb94-4e0f-ac9a-d31b70eaeb77](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/5cac8a05-8211-48db-bf1a-064f871e3e97)
```python
import matplotlib.pyplot as plt
fig, ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```
![311416701-a5473432-41ab-4722-872e-53037544fdeb](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/33e71a35-3cfe-4522-a5e6-b8f2681fa2ef)

```python
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![311416730-7b72bb3b-6824-479c-8703-3d5378b2023e](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/c3faefad-7859-4884-9ecd-9fb0d7a8710f)

```python
#co-relation
import seaborn as sns
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![311416761-6b0e3e5a-85cc-489e-92dc-235451d57476](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/2f9a5ccf-3730-4dc3-a137-74a684e73fe3)

```python
sns.pairplot(dt)
```
![311416795-eb83a5af-b99a-41fd-9c84-e0c828434964](https://github.com/GANESH23012861/EXNO2DS/assets/147139861/b6950c54-4c26-4de2-a78d-1727c1a1f015)

# RESULT
The code successfully excuted
