"""
streaming_pipeline.py - Code for Kafka consumer and Spark Structured Streaming
Handles real-time inference on streaming text data
"""

import json
import os
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import PipelineModel


# ============================================================================
# Stream Configuration
# ============================================================================

def get_stream_schema() -> T.StructType:
    """
    Define schema for streaming input data.
    
    Returns:
        StructType schema for JSON input
    """
    return T.StructType([
        T.StructField("text", T.StringType()),
        T.StructField("label", T.IntegerType()),
    ])


# ============================================================================
# Stream Processing Functions
# ============================================================================

def create_streaming_dataframe(spark: SparkSession, source_path: str,
                               max_files_per_trigger: int = 5) -> DataFrame:
    """
    Create a streaming DataFrame from a file source.
    
    Args:
        spark: SparkSession
        source_path: Path to directory containing JSON files
        max_files_per_trigger: Maximum files to process per trigger
        
    Returns:
        Streaming DataFrame
    """
    schema = get_stream_schema()
    
    stream_df = (spark.readStream
                 .schema(schema)
                 .option("maxFilesPerTrigger", max_files_per_trigger)
                 .json(source_path))
    
    return stream_df


def create_kafka_streaming_dataframe(spark: SparkSession, 
                                     bootstrap_servers: str,
                                     topics: str,
                                     starting_offset: str = "latest") -> DataFrame:
    """
    Create a streaming DataFrame from Kafka source.
    
    Args:
        spark: SparkSession
        bootstrap_servers: Kafka bootstrap servers (e.g., "localhost:9092")
        topics: Comma-separated list of Kafka topics
        starting_offset: Starting offset ("latest", "earliest")
        
    Returns:
        Streaming DataFrame
    """
    stream_df = (spark.readStream
                 .format("kafka")
                 .option("kafka.bootstrap.servers", bootstrap_servers)
                 .option("subscribe", topics)
                 .option("startingOffsets", starting_offset)
                 .load()
                 .selectExpr("CAST(value AS STRING) as json")
                 .select(F.from_json(F.col("json"), get_stream_schema()).alias("data"))
                 .select("data.*"))
    
    return stream_df


# ============================================================================
# Stream Processing Pipeline
# ============================================================================

def apply_stream_processing(stream_df: DataFrame, 
                           feature_pipeline: PipelineModel,
                           model,
                           normalize_udf, tokenize_udf,
                           f011_udf, f034_udf,
                           feature_f011: str, feature_f034: str) -> DataFrame:
    """
    Apply preprocessing and inference to streaming data.
    
    Args:
        stream_df: Streaming DataFrame
        feature_pipeline: Fitted feature pipeline
        model: Trained classification model
        normalize_udf, tokenize_udf: Preprocessing UDFs
        f011_udf, f034_udf: Feature UDFs
        feature_f011, feature_f034: Feature column names
        
    Returns:
        Streaming DataFrame with predictions
    """
    # Clean and preprocess
    stream_clean = (stream_df
        .withColumn("clean_text_columns", normalize_udf(F.col("text")))
        .withColumn("tokens", tokenize_udf(F.col("clean_text_columns")))
        .withColumn("n_tokens", F.size("tokens"))
        .filter(F.col("n_tokens") >= 5)
        .withColumn(feature_f011, f011_udf(F.col("tokens")))
        .withColumn(feature_f034, f034_udf(F.col("text"))))
    
    # Apply feature pipeline
    stream_features = feature_pipeline.transform(stream_clean)
    
    # Apply model
    stream_predictions = (model.transform(stream_features)
                          .select("text", "label", "prediction", "probability"))
    
    return stream_predictions


# ============================================================================
# Stream Output Sinks
# ============================================================================

def write_stream_to_json(stream_df: DataFrame, output_path: str, 
                         checkpoint_path: str, 
                         trigger_interval: str = "3 seconds"):
    """
    Write streaming results to JSON files.
    
    Args:
        stream_df: Streaming DataFrame
        output_path: Output directory path
        checkpoint_path: Checkpoint directory path
        trigger_interval: Processing trigger interval
        
    Returns:
        StreamingQuery object
    """
    query = (stream_df.writeStream
             .outputMode("append")
             .format("json")
             .option("path", output_path)
             .option("checkpointLocation", checkpoint_path)
             .trigger(processingTime=trigger_interval)
             .start())
    
    return query


def write_stream_to_console(stream_df: DataFrame, 
                            trigger_interval: str = "3 seconds"):
    """
    Write streaming results to console for debugging.
    
    Args:
        stream_df: Streaming DataFrame
        trigger_interval: Processing trigger interval
        
    Returns:
        StreamingQuery object
    """
    query = (stream_df.writeStream
             .outputMode("append")
             .format("console")
             .trigger(processingTime=trigger_interval)
             .start())
    
    return query


def write_stream_to_kafka(stream_df: DataFrame, 
                          bootstrap_servers: str,
                          topic: str):
    """
    Write streaming results to Kafka topic.
    
    Args:
        stream_df: Streaming DataFrame
        bootstrap_servers: Kafka bootstrap servers
        topic: Output Kafka topic
        
    Returns:
        StreamingQuery object
    """
    # Convert prediction to JSON string
    to_kafka = stream_df.select(
        F.to_json(F.struct(
            F.col("text"),
            F.col("label"),
            F.col("prediction")
        )).alias("value")
    )
    
    query = (to_kafka.writeStream
             .format("kafka")
             .option("kafka.bootstrap.servers", bootstrap_servers)
             .option("topic", topic)
             .option("checkpointLocation", "./checkpoint/kafka_output")
             .start())
    
    return query


# ============================================================================
# Producer Utilities (for file-based simulation)
# ============================================================================

def produce_sample_files(incoming_path: str, sample_df, limit: int = 20):
    """
    Write sample data as JSON files for file-based stream simulation.
    
    Args:
        incoming_path: Directory to write JSON files
        sample_df: DataFrame with 'text' and 'label' columns
        limit: Maximum number of files to create
    """
    import glob
    
    # Clear existing files
    for fn in glob.glob(os.path.join(incoming_path, "*.json")):
        os.remove(fn)
    
    sample_pd = sample_df.limit(limit).toPandas()
    
    for i, row in sample_pd.iterrows():
        rec = {"text": row["text"], "label": int(row["label"])}
        with open(os.path.join(incoming_path, f"abstract_{i:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    
    print(f"Wrote {len(sample_pd)} JSON files → {incoming_path}")


# ============================================================================
# Main Stream Execution
# ============================================================================

def run_streaming_pipeline(spark: SparkSession, config: dict,
                          feature_pipeline: PipelineModel,
                          model,
                          normalize_udf, tokenize_udf,
                          f011_udf, f034_udf,
                          timeout_seconds: int = 20) -> bool:
    """
    Execute the complete streaming pipeline.
    
    Args:
        spark: SparkSession
        config: Configuration with paths
        feature_pipeline: Fitted feature pipeline
        model: Trained model
        normalize_udf, tokenize_udf: Preprocessing UDFs
        f011_udf, f034_udf: Feature UDFs
        timeout_seconds: Maximum time to run the query
        
    Returns:
        True if completed successfully
    """
    from utils import get_feature_names
    
    incoming_path = config["stream_in"]
    output_path = config["stream_out"]
    checkpoint_path = config["checkpoint_path"]
    
    # Create streaming DataFrame
    stream_df = create_streaming_dataframe(spark, incoming_path)
    
    # Get feature names
    f011_name, f034_name = get_feature_names()
    
    # Apply processing
    predictions = apply_stream_processing(
        stream_df, feature_pipeline, model,
        normalize_udf, tokenize_udf, f011_udf, f034_udf,
        f011_name, f034_name
    )
    
    # Write to JSON output
    query = write_stream_to_json(predictions, output_path, checkpoint_path)
    
    print("Streaming query started. Waiting for data...")
    query.awaitTermination(timeout_seconds)
    query.stop()
    print("Streaming query stopped.")
    
    return True


def evaluate_stream_output(output_path: str, spark: SparkSession) -> Tuple[DataFrame, float]:
    """
    Evaluate streaming output accuracy.
    
    Args:
        output_path: Path to streaming output JSON files
        spark: SparkSession
        
    Returns:
        Tuple of (DataFrame with results, accuracy)
    """
    import glob
    
    pred_files = glob.glob(os.path.join(output_path, "*.json"))
    print(f"Output files written: {len(pred_files)}")
    
    if pred_files:
        results_df = spark.read.json(pred_files)
        results_df.select("text", "label", "prediction").show(5, truncate=60)
        
        pdf = results_df.select("label", "prediction").toPandas()
        accuracy = (pdf.label == pdf.prediction).mean()
        print(f"\nStreaming accuracy: {accuracy:.4f}")
        
        return results_df, accuracy
    
    return None, 0.0