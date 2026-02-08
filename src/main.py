print("RUNNING:", __file__)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

from .preprocessing import preprocessor, preprocessor_dense
from .load_data import load_cleaned_data
from .config import Target, Test_Size, Random_State



models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor()),
        ("model", LogisticRegression(max_iter =5000, solver ="saga", n_jobs =-1,
                                     class_weight ="balanced", random_state =42)),
    ]),
    "LinearSVC": Pipeline([
        ("preprocessor", preprocessor()),
        ("model", LinearSVC(dual="auto",max_iter= 20000, C=1.0, class_weight ="balanced", random_state= 42)),
    ]),
    "ExtraTreesClassifier": Pipeline([
        ("preprocessor", preprocessor_dense()),
        ("model", ExtraTreesClassifier(n_estimators =600, n_jobs =-1,
                                       random_state =42, class_weight ="balanced")),
    ]),
}


def main():
    print("RUNNING MAIN FROM:", __file__)
    df = load_cleaned_data()
    X = df.drop(columns=[Target])
    y = df[Target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =Test_Size, random_state = Random_State, stratify=y)

    pipe = Pipeline([   
        ("preprocessor", preprocessor()),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print("\n" + " " + "="*20 +  name + " " + "="*20)
        print(classification_report(y_test, pred))

if __name__ == "__main__":
    main()