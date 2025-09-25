import os
import yaml
import joblib
import pyspark
import logging
from dotenv import load_dotenv
from pyspark.shell import sc
from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, dayofmonth, lit, from_json, hour, col, when, coalesce
from pyspark.sql.pandas.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType, TimestampType

print("PYSPARK VERSION: ", pyspark.__version__)
print("SPARK VERSION", sc.version)  # Spark version
print(sc._jvm.scala.util.Properties.versionString())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

class FraudDetectionInference:
    bootstrap_servers = None
    topic = None
    security_protocol = None
    sasl_mechanism = None
    username = None
    password = None
    sasl_jaa_config = None

    def __init__(self, config_path="/app/config.yaml"):
        load_dotenv(dotenv_path="/app/.env")
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
        self.model = self._load_model(self.config["model"]["path"])
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        logger.debug("Environment variables loaded: %s", dict(os.environ))

    def _load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
            return model
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

    @staticmethod
    def _load_config(config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _init_spark_session(self):
        try:
            packages = self.config.get("spark").get("packages")
            builder = SparkSession.builder.appName(self.config.get("spark").get("app_name", "FraudDetectionInf"))
            if packages:
                builder = builder.config("spark.jars.packages", packages)
            spark = builder.getOrCreate()
            logger.info(f"Spark Session Initialized: {spark}")
            return spark
        except Exception as e:
            logger.error(f"Error initialising spark session: {str(e)}")
            raise

    def read_from_kafka(self):
        logger.info("Reading data from kafka topic %s", self.config["kafka"]["topic"])
        kafka_config = self.config["kafka"]
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
        kafka_topic = kafka_config["topic"]
        kafka_security_protocol = kafka_config.get("security_protocol", "SASL_SSL")
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username")
        kafka_password = kafka_config.get("password")
        kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.plain.PlainLoginModule required '
            f'username="{kafka_username}" password="{kafka_password}";'
        )

        self.bootstrap_servers = kafka_bootstrap_servers
        self.topic = kafka_topic
        self.security_protocol = kafka_security_protocol
        self.sasl_mechanism = kafka_sasl_mechanism
        self.username = kafka_username
        self.password = kafka_password
        self.sasl_jaa_config = kafka_sasl_jaas_config

        df = (self.spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
                .option("subscribe", kafka_topic)
                .option("startingOffsets", "latest")
                .option("kafka.security.protocol", kafka_security_protocol)
                .option("kafka.sasl.mechanism", kafka_sasl_mechanism)
                .option("kafka.sasl.jaas.config", kafka_sasl_jaas_config)
                .load()
              )

        json_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("location", StringType(), True),
        ])

        parse_df = (df.selectExpr("CAST(value AS STRING)") \
                    .select(from_json(col("value"), json_schema).alias("data"))
                    .select("data.*")
                    )

        return parse_df

    def add_features(self, df):
        # Compute transaction_hour using the timestamp column
        df = df.withColumn("transaction_hour", hour(col('timestamp')))

        # Flag transactions happening at night (between 10 pm and 5am)
        df = df.withColumn("is_night",
                           when((col("transaction_hour") >= 22) | (col("transaction_hour") < 5), 1).otherwise(0))


        df = df.withColumn("is_weekend",
                           when((dayofweek(col("timestamp")) == 1) | (dayofweek(col("timestamp")) == 7), 1).otherwise(0))

        # Transaction day: using day of month
        df = df.withColumn("transaction_day", dayofmonth(col("timestamp")))

        df = df.withColumn("time_since_last_txn", lit(0.0))

        df = df.withColumn("user_activity_24h", lit(1000))
        df = df.withColumn("rolling_avg_7d", lit(1000.0))
        df = df.withColumn("amount_to_avg_ratio", col("amount") / col("rolling_avg_7d"))
        df = df.withColumn("amount_to_avg_ratio", coalesce(col("amount_to_avg_ratio"), lit(1.0)))
        high_risk_merchants = self.config.get("high_risk_merchants", ["QuickCash", "GlobalDigital", "FastMoneyX"])
        df = df.withColumn("merchant_risk", col("merchant").isin(high_risk_merchants).cast("int"))

        df.printSchema()
        return df

    def run_inference(self):
        import pandas as pd
        df = self.read_from_kafka()
        df = df.withWatermark("timestamp", "24 hours")
        feature_df = self.add_features(df)

        broadcast_model = self.broadcast_model

        @pandas_udf("int", PandasUDFType.SCALAR)
        def predict_udf(
                user_id: pd.Series,
                amount: pd.Series,
                currency: pd.Series,
                transaction_hour: pd.Series,
                is_weekend: pd.Series,
                time_since_last_txn: pd.Series,
                merchant_risk: pd.Series,
                amount_to_avg_ratio: pd.Series,
                is_night: pd.Series,
                transaction_day: pd.Series,
                user_activity_24h: pd.Series,
                merchant: pd.Series
        ) -> pd.Series:
            import pandas as pd
            input_df = pd.DataFrame({
                "user_id": user_id,
                "amount": amount,
                "currency": currency,
                "transaction_hour": transaction_hour,
                "is_weekend": is_weekend,
                "time_since_last_txn": time_since_last_txn,
                "merchant_risk": merchant_risk,
                "amount_to_avg_ratio": amount_to_avg_ratio,
                "is_night": is_night,
                "transaction_day": transaction_day,
                "user_activity_24h": user_activity_24h,
                "merchant": merchant
            })

            # Get probabilities of the fraud cases (1)
            prob = broadcast_model.value.predict_proba(input_df)[:, 1]
            threshold = 0.60
            predictions = (prob >= threshold).astype(int)

            return pd.Series(predictions)

        prediction_df = feature_df.withColumn("prediction", predict_udf(
            *[col(f) for f in [
                "user_id", "amount", "currency", "transaction_hour",
                "is_weekend", "time_since_last_txn", "merchant_risk",
                "amount_to_avg_ratio", "is_night", "transaction_day",
                "user_activity_24h", "merchant"
            ]]
        ))

        fraud_predictions = prediction_df.filter(col("prediction") == 1)
        (fraud_predictions.selectExpr(
            "CAST(transaction_id AS STRING) AS key",
            "to_json(struct(*)) AS value"  # Serialize all fields as JSON
        )
         .writeStream
         .format("kafka")
         .option("kafka.bootstrap.servers", self.bootstrap_servers)
         .option("topic", 'fraud_predictions')  # Output topic for fraud alerts
         .option("kafka.security.protocol", self.security_protocol)
         .option("kafka.sasl.mechanism", self.sasl_mechanism)
         .option("kafka.sasl.jaas.config", self.sasl_jaa_config)
         .option("checkpointLocation", "checkpoints/checkpoint")  # For fault tolerance and recovery
         .outputMode("update")  # Only write updated records
         .start()
         .awaitTermination())

if __name__ == "__main__":
    inference = FraudDetectionInference("/app/config.yaml")
    inference.run_inference()