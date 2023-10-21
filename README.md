# Ex-06 FEATURE TRANSFORMATION
### Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
### Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.
### Program:
- Import libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
- Basic Information:
```
df.head()
df.info()
df.info()
```
### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/be5882db-9363-4c7d-83ca-9ac8eecc77d2)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/1e374d35-9db2-4d4c-abf7-460e041e1889)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/579ae4ba-4319-413f-8d5b-36da644a8adc)


### Program:

- Before Transformation:

```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/4625d498-3b56-4c74-ad94-946bdfb82caf)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/50e8990e-531b-40fc-b95e-9fe6529d06a3)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/ff0f1b87-8e9d-4b90-9638-e95486b39e9f)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/4194cca1-d54d-43be-8594-d3f950bdc504)

### Program:

-Log Transformation:

```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```

### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/10836db6-5875-4301-bf47-4171af631ab9)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/9cf74773-4b19-4223-aa94-2e1415f3f03d)

### Program:

- Reciprocal Transformation:

```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/ad57bea2-3d89-4c1b-8455-2912ba71ea50)

### Program:

- SquareRoot Transformation:

```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```

### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/6bcad102-1050-49c3-8f1a-05c5723541ae)

### Program:

- Power Transformation:

```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```

### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/4a8d02d2-d993-43da-ae1f-732000a442ed)


![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/3a4e1dd1-a367-492b-8a8d-5441be52400a)


### Program:

- Quantile Transformation:

```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```

### Output:

![image](https://github.com/22002102/ODD2023-Datascience-Ex06/assets/119091638/367d706f-50b2-47a5-b231-aa654c8d2275)


### Result:
Thus feature transformation is done for the given dataset.
