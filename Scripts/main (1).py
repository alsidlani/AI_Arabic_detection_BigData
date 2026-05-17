"""
main.py - Main orchestration script for Arabic AI-text detection pipeline
Coordinates data preparation, model training, and streaming inference
"""

import os
import sys
import argparse
from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "ArabicAIDetection") -> SparkSession:
    """Create and configure Spark session."""
    spark = (SparkSession.builder
             .appName(app_name)
             .master("local[*]")
             .config("spark.d