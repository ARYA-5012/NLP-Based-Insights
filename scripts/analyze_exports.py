"""
Script to analyze the Kaggle export locally (eda.py).
"""
import pandas as pd
import plotly.express as px
import os

def analyze_exports(data_dir="data/kaggle_exports"):
    """
    Analyze the exported CSVs from Kaggle:
    - Risk labels
    - Confidence scores
    - Competitor mentions
    """
    print("--- Risk Analysis ---")
    risk_path = os.path.join(data_dir, "risk_labels.csv")
    if os.path.exists(risk_path):
        df_risk = pd.read_csv(risk_path)
        print(df_risk["risk_category"].value_counts())
        
    print("\n--- Confidence Trends ---")
    conf_path = os.path.join(data_dir, "qa_alpha_scores.csv")
    if os.path.exists(conf_path):
        df_conf = pd.read_csv(conf_path)
        print(df_conf.head())

    print("\n--- Competitor Mentions ---")
    comp_path = os.path.join(data_dir, "competitor_mentions.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        print(df_comp["competitor"].value_counts())

if __name__ == "__main__":
    analyze_exports()
