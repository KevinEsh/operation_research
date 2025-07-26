import os

import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.linear_model import LinearRegression

os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlflow123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:10000"
os.environ["AWS_REGION"] = "us-east-1"


X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

mlflow.set_tracking_uri("http://localhost:5000")

has_exp = mlflow.get_experiment_by_name("linear_regression")
# If the experiment does not exist, create it
if has_exp is None:
    experiment_id = mlflow.create_experiment("linear_regression")
else:
    experiment_id = has_exp.experiment_id
# experiment_id = mlflow.create_experiment("linear_regression_example")

# x_train, y_train = load_pickle(train_bundle_path)

mlflow.sklearn.autolog(log_datasets=False, log_models=True)

with mlflow.start_run(experiment_id=experiment_id) as run:
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # mlflow.sklearn.log_model(scaler, artifact_path="scaler")
    for _ in range(3):
        reg = LinearRegression().fit(X, y)
        reg.score(X, y)
        reg.predict(np.array([[3, 5]]))
