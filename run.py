from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from dataset import *
from pipeline import *


data = load_data()
X_train, X_test, y_train, y_test = data_split(data)

classifiers = [
    LogisticRegression(), SVC(), KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()
]

pipeline = define_pipeline()
run_pipeline(pipeline, classifiers, X_train, y_train)
