from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


def define_pipeline():
    pipeline = Pipeline([
        ('normalizer', StandardScaler()), # step1 - normalize data
        ('classifier', LogisticRegression()) #step2 - classify
    ])
    
    # view pipeline steps
    print(pipeline.steps)
    return pipeline


def run_pipeline(pipeline, classifiers, X_train, y_train):
    for classifier in classifiers:
        pipeline.set_params(classifier = classifier)
        scores = cross_validate(pipeline, X_train, y_train)
        print('---------------------------------')
        print(str(classifier))
        print(f'accuracy: {scores["test_score"].mean()}')