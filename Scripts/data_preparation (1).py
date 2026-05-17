"""
data_preparation.py - Distributed functions for data cleaning and feature engineering
Handles Spark DataFrame transformations, Arabic text preprocessing, and feature extraction
"""

import os
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, VectorAssembler, StandardScaler

# Import from utils
from utils import (
    normalize_arabic, 
    tokenize_arabic, 
    compute_long_word_ratio,
    count_total_lines,
    get_feature_names
)


# ============================================================================
# Spark UDF Wrappers
# ============================================================================

def create_udfs(spark_context, stopwords_set):
    """
    Create Spark UDFs for Arabic text processing.
    
    Args:
        spark_context: SparkContext for broadcasting
        stopwords_set: Set of Arabic stopwords
        
    Returns:
        Tuple of UDFs (normalize_udf, tokenize_udf, f011_udf, f034_udf, bigrams_udf)
    """
    
    # Broadcast stopwords for efficient distribution
    sw_bc = spark_context.broadcast(stopwords_set)
    
    # Normalization UDF
    normalize_udf = F.udf(normalize_arabic, T.StringType())
    
    # Tokenization UDF with stopword filtering
    def _tokenize(text):
        return tokenize_arabic(text, sw_bc.value)
    
    tokenize_udf = F.udf(_tokenize, T.ArrayType(T.StringType()))
    
    # f011 - Long word ratio UDF
    f011_udf = F.udf(lambda ws: compute_long_word_ratio(ws), T.DoubleType())
    
    # f034 - Line count UDF
    f034_udf = F.udf(lambda txt: count_total_lines(txt), T.IntegerType())
    
    # Bigram generation UDF
    def _to_bigrams(toks):
        return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    
    bigrams_udf = F.udf(_to_bigrams, T.ArrayType(T.StringType()))
    
    return normalize_udf, tokenize_udf, f011_udf, f034_udf, bigrams_udf


# ============================================================================
# Data Cleaning
# ============================================================================

def clean_dataframe(df: DataFrame, normalize_udf, tokenize_udf, 
                    min_tokens: int = 5) -> DataFrame:
    """
    Apply cleaning and tokenization to the input DataFrame.
    
    Args:
        df: Input Spark DataFrame with 'text' column
        normalize_udf: UDF for Arabic normalization
        tokenize_udf: UDF for tokenization
        min_tokens: Minimum tokens required to keep a row
        
    Returns:
        Cleaned DataFrame with cleaned text, tokens, and token count
    """
    df_clean = (
        df
        .withColumn("clean_text_columns", normalize_udf(F.col("text")))
        .withColumn("tokens", tokenize_udf(F.col("clean_text_columns")))
        .withColumn("n_tokens", F.size("tokens"))
        .filter(F.col("n_tokens") >= min_tokens)
    )
    
    return df_clean.cache()


# ============================================================================
# Feature Engineering Pipeline
# ============================================================================

def build_feature_pipeline(feature_f011: str, feature_f034: str) -> Pipeline:
    """
    Build the complete feature engineering pipeline.
    
    Args:
        feature_f011: Name of f011 feature column
        feature_f034: Name of f034 feature column
        
    Returns:
        Spark ML Pipeline object
    """
    hashing = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=4096)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    
    stylo_assembler = VectorAssembler(
        inputCols=[feature_f011, feature_f034, "n_tokens"],
        outputCol="stylo_raw"
    )
    
    scaler = StandardScaler(
        inputCol="stylo_raw", 
        outputCol="stylo",
        withMean=False, 
        withStd=True
    )
    
    final_assembler = VectorAssembler(
        inputCols=["tfidf", "stylo"], 
        outputCol="features"
    )
    
    return Pipeline(stages=[hashing, idf, stylo_assembler, scaler, final_assembler])


def prepare_features(df: DataFrame, pipeline: Pipeline) -> DataFrame:
    """
    Apply feature engineering pipeline to the DataFrame.
    
    Args:
        df: Cleaned Spark DataFrame with 'tokens' column
        pipeline: Fitted or unfitted Pipeline object
        
    Returns:
        DataFrame with 'features' column
    """
    if isinstance(pipeline, Pipeline):
        pipeline_model = pipeline.fit(df)
    else:
        pipeline_model = pipeline
    
    df_ready = pipeline_model.transform(df).select("features", "label").cache()
    return df_ready, pipeline_model


def add_stylometric_features(df: DataFrame, feature_f011: str, feature_f034: str,
                            f011_udf, f034_udf) -> DataFrame:
    """
    Add stylometric features to the DataFrame.
    
    Args:
        df: DataFrame with 'tokens' and 'text' columns
        feature_f011: Name for f011 feature column
        feature_f034: Name for f034 feature column
        f011_udf: UDF for computing long word ratio
        f034_udf: UDF for counting lines
        
    Returns:
        DataFrame with added feature columns
    """
    df_feat = (
        df
        .withColumn(feature_f011, f011_udf(F.col("tokens")))
        .withColumn(feature_f034, f034_udf(F.col("text")))
    ).cache()
    
    return df_feat


# ============================================================================
# Data Persistence
# ============================================================================

def save_parquet(df: DataFrame, output_path: str):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: Spark DataFrame to save
        output_path: Destination path
    """
    df.select("text", "clean_text_columns", "tokens", "n_tokens",
              "label", "model", "source") \
      .write.mode("overwrite").parquet(output_path)


def load_parquet(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load DataFrame from Parquet format.
    
    Args:
        spark: SparkSession
        input_path: Source path
        
    Returns:
        Loaded Spark DataFrame
    """
    return spark.read.parquet(input_path).cache()


# ============================================================================
# Main Execution Function
# ============================================================================

def run_data_preparation(spark: SparkSession, config: dict) -> DataFrame:
    """
    Execute the complete data preparation pipeline.
    
    Args:
        spark: SparkSession
        config: Configuration dictionary with paths and settings
        
    Returns:
        Prepared DataFrame with features
    """
    from utils import get_feature_names, load_and_melt_dataset
    import pandas as pd
    
    # Load and melt dataset
    raw_path = config["raw_data_path"]
    ai_cols = ['allam_generated_abstract', 'jais_generated_abstract',
               'llama_generated_abstract', 'openai_generated_abstract']
    
    long_pd = load_and_melt_dataset(raw_path, ai_cols)
    
    schema = T.StructType([
        T.StructField("text", T.StringType(), False),
        T.StructField("label", T.IntegerType(), False),
        T.StructField("model", T.StringType(), False),
        T.StructField("source", T.StringType(), False),
    ])
    
    df = spark.createDataFrame(long_pd, schema=schema).cache()
    
    # Load stopwords
    from nltk.corpus import stopwords as nltk_stopwords
    ar_stopwords = set(nltk_stopwords.words("arabic"))
    
    # Create UDFs
    normalize_udf, tokenize_udf, f011_udf, f034_udf, bigrams_udf = create_udfs(
        spark.sparkContext, ar_stopwords
    )
    
    # Clean data
    df_clean = clean_dataframe(df, normalize_udf, tokenize_udf)
    
    # Save intermediate Parquet
    proc_path = config.get("proc_path", "./data/processed/abstracts.parquet")
    save_parquet(df_clean, proc_path)
    
    # Load back
    df_proc = load_parquet(spark, proc_path)
    
    # Add stylometric features
    f011_name, f034_name = get_feature_names()
    df_feat = add_stylometric_features(df_proc, f011_name, f034_name, 
                                       f011_udf, f034_udf)
    
    # Build and apply feature pipeline
    feature_pipeline = build_feature_pipeline(f011_name, f034_name)
    df_ready, pipeline_model = prepare_features(df_feat, feature_pipeline)
    
    return df_ready, pipeline_model