# evaluate model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np   
from sklearn.metrics import (accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    classification_report, roc_auc_score
)

def evaluate_model(model_name, y_true, y_pred, roc_score):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate specificity and NPV
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    
    # Get classification report as a dictionary
    report = classification_report(y_true, y_pred, output_dict=True)

    # Extract class-specific metrics
    class_0_metrics = report["0.0"]  # Adjust if class labels are different
    class_1_metrics = report["1.0"]

    # Compute overall metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        
        # Class 0 Metrics
        "Precision_0": class_0_metrics["precision"],
        "Recall_0": class_0_metrics["recall"],
        "F1_Score_0": class_0_metrics["f1-score"],
        
        # Class 1 Metrics
        "Precision_1": class_1_metrics["precision"],
        "Recall_1": class_1_metrics["recall"],
        "F1_Score_1": class_1_metrics["f1-score"],

        # Specificity & NPV
        "Specificity": specificity,
        "Negative Predictive Value (NPV)": npv,

        "ROC score": roc_score
    }

    # Print results
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("-" * 50)
    print(f"Class 0:")
    print(f"    Precision: {metrics['Precision_0']:.4f}")
    print(f"    Recall: {metrics['Recall_0']:.4f}")
    print(f"    F1 Score: {metrics['F1_Score_0']:.4f}")
    print(f"Class 1:")
    print(f"    Precision: {metrics['Precision_1']:.4f}")
    print(f"    Recall: {metrics['Recall_1']:.4f}")
    print(f"    F1 Score: {metrics['F1_Score_1']:.4f}")
    print("-" * 50)
    print(f"Specificity: {metrics['Specificity']:.4f}")
    print(f"Negative Predictive Value (NPV): {metrics['Negative Predictive Value (NPV)']:.4f}")
    print(f"ROC score: {metrics['ROC score']:.4f}")
    print("-" * 50)

    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True,  fmt='.2f', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix (KNN)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return metrics

