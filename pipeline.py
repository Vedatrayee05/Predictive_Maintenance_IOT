import re
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb


def load_data(file_path):
    df = pd.read_csv(file_path)
    df_clean = df.drop(columns=['UDI', 'Product ID'])
    df_clean = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)
    df_clean.columns = [re.sub(r'[\[\]<]', '', col) for col in df_clean.columns]

    X = df_clean.drop(
        columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    )
    y = df_clean['Machine failure']
    return X, y


def run_pipeline(X, y):
    # 1. Train-Test Split to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2.creating pipeline (SMOTE will do only training)
    pipeline = ImbPipeline(
        [
            ('smote', SMOTE(random_state=42)),
            (
                'classifier',
                xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            ),
        ]
    )

    # 3. finding best parameter Hyperparameter Tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
    )

    # 4. Model training
    grid_search.fit(X_train, y_train)

    # 5. Model saving
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'xgboost_predictive_model.pkl')
    print("Success: Model trained & saved as 'xgboost_predictive_model.pkl'")

    return best_model


if __name__ == '__main__':
    X, y = load_data('ai4i2020.csv')
    run_pipeline(X, y)