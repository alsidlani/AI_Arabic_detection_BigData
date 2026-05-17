"""
modeling.py - Spark MLlib code for building, training, and evaluating models
Handles classification model training, evaluation, and persistence
"""

import time
from typing import Dict, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression, NaiveBayes, RandomForestClassifier,
    LogisticRegressionModel, NaiveBayesModel, RandomForestClassificationModel
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator
)
from pyspark.sql import functions as F


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIGS = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {"maxIter": 30, "regParam": 0.01}
    },
    "NaiveBayes": {
        "class": NaiveBayes,
        "params": {"smoothing": 1.0, "modelType": "multinomial"}
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {"numTrees": 60, "maxDepth": 10, "seed": 42}
    }
}


# ============================================================================
# Model Training
# ============================================================================

def train_model(train_df: DataFrame, model_name: str, 
                model_params: Dict = None) -> Union[LogisticRegressionModel, 
                                                     NaiveBayesModel, 
                                                     RandomForestClassificationModel]:
    """
    Train a classification model on the training data.
    
    Args:
        train_df: Training DataFrame with 'features' and 'label' columns
        model_name: Name of the model to train ('LogisticRegression', 'NaiveBayes', 'RandomForest')
        model_params: Optional override for default parameters
        
    Returns:
        Trained Spark ML model
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    params = model_params or config["params"]
    model_class = config["class"]
    
    model = model_class(**params)
    trained_model = model.fit(train_df)
    
    return trained_model


def train_with_tracking(train_df: DataFrame, val_df: DataFrame, 
                        model_name: str) -> Tuple[Dict, object]:
    """
    Train a model and return training metrics and time.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        model_name: Name of the model to train
        
    Returns:
        Tuple of (metrics_dict, trained_model)
    """
    start_time = time.time()
    
    model = train_model(train_df, model_name)
    train_time = time.time() - start_time
    
    metrics = evaluate_model(model, val_df, model_name)
    metrics["train_time_sec"] = round(train_time, 2)
    
    return metrics, model


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(model, eval_df: DataFrame, model_name: str = "") -> Dict:
    """
    Evaluate a trained model on a DataFrame.
    
    Args:
        model: Trained Spark ML model
        eval_df: Evaluation DataFrame with 'features' and 'label' columns
        model_name: Name prefix for logging
        
    Returns:
        Dictionary with evaluation metrics (accuracy, f1, auc)
    """
    predictions = model.transform(eval_df)
    
    acc_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(metricName="f1")
    auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    
    accuracy = acc_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "auc": round(auc, 4),
        "predictions": predictions
    }
    
    if model_name:
        print(f"{model_name:<22s}  acc={accuracy:.4f}  f1={f1_score:.4f}  AUC={auc:.4f}")
    
    return metrics


def get_confusion_matrix(predictions: DataFrame) -> DataFrame:
    """
    Generate confusion matrix from model predictions.
    
    Args:
        predictions: DataFrame with 'label' and 'prediction' columns
        
    Returns:
        DataFrame with confusion matrix counts
    """
    cm = (predictions
          .groupBy("label", "prediction")
          .count()
          .toPandas()
          .pivot(index="label", columns="prediction", values="count")
          .fillna(0)
          .astype(int))
    
    return cm


# ============================================================================
# Model Persistence
# ============================================================================

def save_model(model, model_path: str):
    """
    Save trained model to disk.
    
    Args:
        model: Trained Spark ML model
        model_path: Destination path
    """
    import shutil
    shutil.rmtree(model_path, ignore_errors=True)
    model.write().overwrite().save(model_path)


def load_model(model_path: str):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    # Determine model type from path (simplified - in production, you'd save model info)
    from pyspark.ml.classification import LogisticRegressionModel
    
    return LogisticRegressionModel.load(model_path)


def save_pipeline(pipeline_model, pipeline_path: str):
    """
    Save feature pipeline to disk.
    
    Args:
        pipeline_model: Fitted PipelineModel
        pipeline_path: Destination path
    """
    import shutil
    shutil.rmtree(pipeline_path, ignore_errors=True)
    pipeline_model.write().overwrite().save(pipeline_path)


# ============================================================================
# Model Selection
# ============================================================================

def select_best_model(results: Dict[str, Dict], metric: str = "f1_score") -> str:
    """
    Select the best model based on a specified metric.
    
    Args:
        results: Dictionary of model results
        metric: Metric to use for selection ('accuracy', 'f1_score', 'auc')
        
    Returns:
        Name of the best model
    """
    best_name = max(results, key=lambda k: results[k].get(metric, 0))
    return best_name


def print_results_summary(results: Dict[str, Dict]):
    """
    Print formatted summary of all model results.
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<10} {'Train Time (s)':<15}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<18} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
              f"{metrics['auc']:<10.4f} {metrics.get('train_time_sec', 0):<15.2f}")
    
    print("=" * 70)


# ============================================================================
# Feature Importance (Random Forest)
# ============================================================================

def get_feature_importance(rf_model, feature_names: list) -> Dict[str, float]:
    """
    Extract feature importance from a trained Random Forest model.
    
    Args:
        rf_model: Trained RandomForestClassificationModel
        feature_names: List of feature names in order
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not isinstance(rf_model, RandomForestClassificationModel):
        raise ValueError("Feature importance is only available for RandomForest models")
    
    importances = rf_model.featureImportances.toArray()
    
    # For this pipeline, stylometric features are the last 3 dimensions
    stylo_importance = importances[-3:]
    stylo_labels = feature_names
    
    return dict(zip(stylo_labels, [round(float(x), 5) for x in stylo_importance]))


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_modeling_pipeline(train_df: DataFrame, val_df: DataFrame, 
                          test_df: DataFrame) -> Tuple[Dict, object]:
    """
    Run complete modeling pipeline: train multiple models, evaluate, select best.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (results_dict, best_model)
    """
    results = {}
    trained_models = {}
    
    # Train each model
    for model_name in MODEL_CONFIGS.keys():
        print(f"\nTraining {model_name}...")
        metrics, model = train_with_tracking(train_df, val_df, model_name)
        results[model_name] = metrics
        trained_models[model_name] = model
    
    # Print summary
    print_results_summary(results)
    
    # Select best model
    best_name = select_best_model(results)
    best_model = trained_models[best_name]
    
    # Evaluate on test set
    print(f"\nEvaluating best model ({best_name}) on test set...")
    test_metrics = evaluate_model(best_model, test_df, f"{best_name} (TEST)")
    results[best_name]["test_metrics"] = test_metrics
    
    return results, best_model