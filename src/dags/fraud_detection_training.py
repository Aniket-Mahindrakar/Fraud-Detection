import os
import yaml
import boto3
import mlflow
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, average_precision_score, precision_score, \
    f1_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler("./fraud_detection_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionTraining:
    def __init__(self, config_path='/app/config.yaml'):
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/usr/bin/git"

        load_dotenv(dotenv_path="/app/.env")

        self.config = self.load_config(config_path)

        os.environ.update({
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_S3_ENDPOINT_URL": self.config["mlflow"]["s3_endpoint_url"],
        })

        self._validate_environment()
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration file: %s", str(e))
            raise

    def _validate_environment(self):
        required_vars = ["KAFKA_BOOTSTRAP_SERVERS", "KAFKA_USERNAME", "KAFKA_PASSWORD"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

        self._check_minio_connection()

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=self.config["mlflow"]["s3_endpoint_url"],
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )

            buckets = s3.list_bucket()
            bucket_names = [b["Name"] for b in buckets.get("Buckets", [])]
            logger.info(f"Minio connection verified. Buckets: {bucket_names}")

            mlflow_bucket = self.config["mlflow"].get("bucket", "mlflow")

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info("Created missing MLFlow bucket: %s", mlflow_bucket)

        except Exception as e:
            logger.error(f"Minio connection failed: {str(e)}")

    def read_from_kafka(self):
        try:
            topic = self.config["kafka"]["topic"]
            logger.info(f"Connecting to Kafka topic: {topic}")

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config["kafka"]["bootstrap_servers"].split(","),
                security_protocol="SASL_SSL",
                sasl_mechanism="PLAINTEXT",
                sasl_plain_username=self.config["kafka"]["username"],
                sasl_plain_password=self.config["kafka"]["password"],
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
                consumer_timeout_ms=self.config["kafka"].get("timeout", 10000),
            )

            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)
            if(df.empty):
                raise ValueError("No messages received from Kafka")

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            if("is_fraud" not in df.columns):
                raise ValueError("Fraud label (is_fraud) is missing from Kafka data")

            # Calculate and log the fraud rate (percentage of fraudulent transactions)
            fraud_rate = df["is_fraud"].mean() * 100
            logger.info("Kafka data read successfully with fraud rate: %.2f%%", fraud_rate)

            return df
        except Exception as e:
            logger.error("Failed to read data from Kafka: %s", str(e), exc_info=True)
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["user_id", "timestamp"]).copy()

        # ----- Temporal Features -----
        # Hour of the transaction
        df["transaction_hour"] = df["timestamp"].dt.hour

        # Flag transactions happening at night
        df["is_night"] = ((df["transaction_hour"] >= 22) | (df["transaction_hour"] < 5)).astype(int)

        # Flag transactions happening on weekend (Saturday=5 and Sunday=6)
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

        # Transaction Day
        df["transaction_day"] = df["timestamp"].dt.day

        # ----- Behavioral Features -----
        df["user_activity_24h"] = df.groupby("user_id", group_keys=False).apply(
            lambda x: x.rolling("24h", on="timestamp", closed="left")["amount"].count().fillna(0)
        )

        # ----- Monetary Features -----
        df["amount_to_avg_ratio"] = df.groupby("user_id", group_keys=False).apply(
            lambda g: (g["amount"] / g["amount"].rolling(7, min_periods=1).mean()).fillna(1.0)
        )

        # ----- Merchant Features -----
        high_risk_merchants = self.config.get("high_risk_merchants", ["QuickCash", "GlobalDigital", "FastMoneyX"])
        df ["merchant_risk"] = df["merchant"].isin(high_risk_merchants).astype(int)

        feature_cols = [
            "amount",
            "is_night",
            "is_weekend",
            "transaction_day",
            "user_activity_24h",
            "amount_to_avg_ratio",
            "merchant_risk",
            "merchant",
        ]

        if("is_fraud" not in df.columns):
            raise ValueError('Missing target column "is_fraud"')

        return df[feature_cols + ["is_fraud"]]

    def train_model(self):
        try:
            logger.info(f"Training model started.")

            # Read raw transactions data from Kafka
            df = self.read_from_kafka()

            # Create Feature Engineering from the raw data
            data = self.create_features(df)

            # Split the data into features (X) and target (Y)
            X = data.drop(columns=["is_fraud"])
            y = data["is_fraud"]

            if(y.sum() == 0):
                raise ValueError("No positive samples exist in training data")

            if(y.sum() < 10):
                logger.warning("Low positive samples: %d. Consider additional data augmentation.", y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["model"].get("test_size", 0.2),
                stratify=y,
                random_state=self.config["model"].get("seed", 42),
            )

            with mlflow.start_run():
                mlflow.log_metrics({
                    "train_samples": X_train.shape[0],
                    "positive_samples": int(y_train.sum()),
                    "class_ratio": float(y_train.mean()),
                    "test_samples": X_test.shape[0],
                })

                preprocessor = ColumnTransformer(
                    [
                        ("merchant_encoder", OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            dtype=np.float32,
                        ), ["merchant"])
                    ], remainder="passthrough"
                )

                xgb = XGBClassifier(
                    eval_metric="aucpr",
                    random_state=self.config["model"].get("seed", 42),
                    reg_lambda=1.0,
                    n_estimators=self.config["model"]["params"]["n_estimators"],
                    n_jobs=1,
                    tree_method=self.config["model"].get("tree_method", "hist"),
                )

                # ----- Pipeline Construction -----
                # Preprocessing
                # Class Imbalance handling with SMOTE
                # Prediction with xgboost classifier
                pipeline = ImbPipeline([
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=self.config["model"].get("seed", 42))),
                    ("classifier", xgb),
                ], memory="./cache")

                params_dist = {
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__learning_rate": [0.01, 0.05, 0.1],
                    "classifier__subsample": [0.6, 0.8, 1.0],
                    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                    "classifier__gamma": [0.0, 0.1, 0.3],
                    "classifier__reg_alpha": [0.0, 0.1, 0.5],
                }

                search = RandomizedSearchCV(
                    pipeline,
                    params_dist,
                    n_iter=20,
                    scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                    random_state=self.config["model"].get("seed", 42),
                )

                logger.info("Starting hyper-parameter tuning...")
                search.fit(X_train, y_train)
                best_model = search.best_estimator_ # Best model pipeline
                best_params = search.best_params_ # Best hyperparameters
                logger.info("Best hyper-parameters: %s", best_params)

                train_proba = best_model.predict_proba(X_train)[:, 1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve((y_train, train_proba))
                f1_scores = [2 * (p * r)/(p + r) if (p + r) > 0 else 0 for p, r in zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info("Optimal threshold determined as: %.4f", best_threshold)

                X_test_processed = best_model.named_steps["preprocessor"].transform(X_test)
                test_proba = best_model.named_steps["classifier"].predict_proba(X_test_processed)[:, 1]
                y_pred = (test_proba >= best_threshold).astype(int)

                metrics = {
                    "auc_pr": float(average_precision_score(y_test, test_proba)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                    "threshold": float(best_threshold),
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=[6, 4])
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion matrix")
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ["Not Fraud", "Fraud"])
                plt.yticks(tick_marks, ["Not Fraud", "Fraud"])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="red")

                plt.tight_layout()
                cm_file_name = "confusion_matrix.png"
                plt.savefig(cm_file_name)
                mlflow.log_artifact(cm_file_name)

                # Precision and Recall Curve
                plt.figure(figsize=[10, 6])
                plt.plot(recall_arr, precision_arr, marker=".", label="Precision-Recall Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend()
                pr_file_name = "precision_recall_curve.png"
                plt.savefig(pr_file_name)
                mlflow.log_artifact(pr_file_name)
                plt.close()

                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name="fraud_detection_model",
                )

                logger.info("Training successfully completed with metrics: %s", metrics)

                return best_model, metrics

        except Exception as e:
            logger.error("Training Failed: %s", str(e), exc_info=True)
            raise