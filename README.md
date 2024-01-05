## Building a Scikit Learn Classification Pipeline
<hr>

This is a basic example of building a classification pipeline of different classifiers using scikit learn. Scikit learn pipeline provides us with a structured framework for applying transformations to the data and performing other necessary machine learning tasks sequentially. It clearly outlines which processing steps we chose to apply, their order, and the exact parameters we applied.

### Read Dataset:


```python
import numpy as np 
import pandas as pd 

data = pd.read_csv('Iris.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



The first five columns in the dataset are features and the last column is the class label. 
The Id column has no use for model training and need to delete it.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             150 non-null    int64  
     1   SepalLengthCm  150 non-null    float64
     2   SepalWidthCm   150 non-null    float64
     3   PetalLengthCm  150 non-null    float64
     4   PetalWidthCm   150 non-null    float64
     5   Species        150 non-null    object 
    dtypes: float64(4), int64(1), object(1)
    memory usage: 7.2+ KB



```python
data.drop('Id',axis=1,inplace=True)
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   SepalLengthCm  150 non-null    float64
     1   SepalWidthCm   150 non-null    float64
     2   PetalLengthCm  150 non-null    float64
     3   PetalWidthCm   150 non-null    float64
     4   Species        150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB


### Encode Label:

The class label in the dataset is categorical. So, we need to apply LabelEncoder in order to encode our categorical class label into numerical values.


```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

"""
LabelEncoder:
LabelEncoder can be used to normalize labels. 
It can also be used to transform non-numerical labels to numerical labels. 
Fit label encoder. Fit label encoder and return encoded labels.

OneHotEncoder:
Encode categorical features as a one-hot numeric array
"""
data['Species'] = LabelEncoder().fit_transform(data['Species'])
data.iloc[[0, -1],:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Define Pipeline:


```python
pipeline = Pipeline([
    ('normalizer', StandardScaler()), # step1 - normalize data
    ('classifier', LogisticRegression()) #step2 - classify
])

# view pipeline steps
pipeline.steps
```




    [('normalizer', StandardScaler()), ('classifier', LogisticRegression())]



### Dataset Split:


```python
X, y = data.values[:, 0:4], data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
```

    X_train shape: (120, 4)
    X_test shape: (30, 4)
    y_train shape: (120,)
    y_test shape: (30,)


### Run Pipeline:


```python
def run_pipeline(classifiers):
    for classifier in classifiers:
        pipeline.set_params(classifier = classifier)
        scores = cross_validate(pipeline, X_train, y_train)
        print('-'*40)
        print(str(classifier))
        print(f'accuracy: {scores["test_score"].mean()}')
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

classifiers = [
    LogisticRegression(), SVC(), KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()
]

run_pipeline(classifiers)
```

    ----------------------------------------
    LogisticRegression()
    accuracy: 0.95
    ----------------------------------------
    SVC()
    accuracy: 0.9416666666666667
    ----------------------------------------
    KNeighborsClassifier(n_neighbors=3)
    accuracy: 0.9166666666666667
    ----------------------------------------
    DecisionTreeClassifier()
    accuracy: 0.8916666666666666
    ----------------------------------------
    RandomForestClassifier()
    accuracy: 0.9333333333333333
    ----------------------------------------
    GradientBoostingClassifier()
    accuracy: 0.9166666666666666

