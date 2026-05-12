from dotenv import load_dotenv

import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/rizkyzhang/Superstore_Sales_Predict_Test.mlflow")
mlflow.set_experiment("superstore-sales")
mlflow.sklearn.autolog(registered_model_name="SuperstoreSalesModel")

df = pd.read_csv("sample_superstore_preprocessing.csv")

X = df.drop(columns=["Sales", "Discount_Bin"])
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LinearRegression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
