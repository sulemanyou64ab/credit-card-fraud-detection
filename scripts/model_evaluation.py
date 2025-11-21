"""
Model evaluation and comparison module
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PLOTS_DIR


def create_results_summary(results_dict):
    """
    Create a summary dataframe of all model results
    
    Args:
        results_dict (dict): Dictionary with model names as keys and AUC scores as values
        
    Returns:
        pd.DataFrame: Summary dataframe
    """
    df = pd.DataFrame(list(results_dict.items()), columns=['Model', 'AUC Score'])
    df = df.sort_values('AUC Score', ascending=False).reset_index(drop=True)
    
    return df


def plot_model_comparison(results_df, save=True):
    """
    Plot comparison of all models
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save (bool): Whether to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='AUC Score', y='Model', palette='viridis')
    plt.title('Model Performance Comparison (ROC-AUC Score)', fontsize=16)
    plt.xlabel('ROC-AUC Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0.75, 1.0)
    
    # Add value labels on bars
    for i, row in results_df.iterrows():
        plt.text(row['AUC Score'] - 0.02, i, f"{row['AUC Score']:.4f}", 
                va='center', fontsize=10, color='white', weight='bold')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_evaluation_report(results_dict):
    """
    Generate a comprehensive evaluation report
    
    Args:
        results_dict (dict): Dictionary with model names as keys and AUC scores as values
        
    Returns:
        str: Formatted report
    """
    results_df = create_results_summary(results_dict)
    
    report = "\n" + "="*70 + "\n"
    report += "MODEL EVALUATION SUMMARY\n"
    report += "="*70 + "\n\n"
    
    report += "Model Performance Rankings:\n"
    report += "-" * 70 + "\n"
    
    for i, row in results_df.iterrows():
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        report += f"{rank_emoji} {i+1}. {row['Model']:<25} AUC: {row['AUC Score']:.4f}\n"
    
    report += "\n" + "="*70 + "\n"
    report += f"Best Model: {results_df.iloc[0]['Model']} with AUC = {results_df.iloc[0]['AUC Score']:.4f}\n"
    report += "="*70 + "\n"
    
    return report, results_df


def save_results_to_csv(results_df, filename='model_results.csv'):
    """
    Save results to CSV file
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        filename (str): Output filename
    """
    output_path = f'{PLOTS_DIR}/{filename}'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
