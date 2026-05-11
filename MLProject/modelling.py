import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dagshub
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

load_dotenv()
dagshub.init(repo_owner="rizkyzhang", repo_name="Superstore_Sales_Predict", mlflow=True)
mlflow.set_experiment("superstore-sales-baseline")

df = pd.read_csv("sample_superstore_preprocessing.csv")

X = df.drop(columns=["Sales", "Discount_Bin"])
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LinearRegression") as run:
    mlflow.sklearn.autolog()

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    mlflow.log_metrics({"rmse": rmse, "mape": mape})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_title("Actual vs Predicted")
    axes[0].set_xlabel("Actual Sales")
    axes[0].set_ylabel("Predicted Sales")

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title("Residuals Plot")
    axes[1].set_xlabel("Predicted Sales")
    axes[1].set_ylabel("Residuals")

    plt.tight_layout()
    plt.savefig("screenshot_artifact.png")
    mlflow.log_artifact("screenshot_artifact.png")
    plt.close()

    mlflow.register_model(f"runs:/{run.info.run_id}/model", "SuperstoreSalesModel")

    print("Done — run logged and model registered to DagsHub")
