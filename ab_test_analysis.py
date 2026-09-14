import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. DATA GENERATION
def generate_ab_test_data(filepath='website_ab_test.csv', n_records_per_group=500):
    """Generates synthetic A/B test data for a website redesign."""
    np.random.seed(42)
    
    # Old Design Data
    old_ratings = np.random.choice([1, 2, 3, 4, 5], size=n_records_per_group, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    old_time = np.random.lognormal(3, 0.5, n_records_per_group)
    old_purchased = np.random.choice([0, 1], size=n_records_per_group, p=[0.7, 0.3]) # 30% conversion
    
    # New Design Data
    new_ratings = np.random.choice([1, 2, 3, 4, 5], size=n_records_per_group, p=[0.05, 0.15, 0.3, 0.3, 0.2])
    new_time = np.random.lognormal(2.8, 0.5, n_records_per_group)
    new_purchased = np.random.choice([0, 1], size=n_records_per_group, p=[0.6, 0.4]) # 40% conversion
    
    # Combine into DataFrame
    df_old = pd.DataFrame({'Design': 'Old', 'Satisfaction': old_ratings, 'TimeOnSite': old_time, 'Purchased': old_purchased})
    df_new = pd.DataFrame({'Design': 'New', 'Satisfaction': new_ratings, 'TimeOnSite': new_time, 'Purchased': new_purchased})
    
    df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(filepath, index=False)
    print(f"[*] Synthetic A/B test data saved to {filepath}")
    return df

# 2. EXPLORATORY DATA ANALYSIS (EDA)
def perform_eda(df):
    """Generates visualizations comparing the old and new website designs."""
    print("\n[*] Running Exploratory Data Analysis...")
    
    plt.figure(figsize=(18, 5))

    # Satisfaction Ratings
    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x='Satisfaction', hue='Design', palette='mako')
    plt.title('User Satisfaction Distribution')

    # Time on Site
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='Design', y='TimeOnSite', hue='Design', palette='mako')
    plt.title('Time on Site (seconds)')
    plt.yscale('log') 
    plt.legend([],[], frameon=False) 

    # Purchase Conversion Rate
    plt.subplot(1, 3, 3)
    # Replaced errorbar=None with ci=None for older Seaborn compatibility
    sns.barplot(data=df, x='Design', y='Purchased', hue='Design', palette='mako', ci=None)
    plt.title('Purchase Conversion Rate')
    plt.ylabel('Conversion Rate')
    plt.legend([],[], frameon=False) 

    plt.tight_layout()
    plt.show()

# 3. STATISTICAL TESTING
def run_statistical_tests(df, alpha=0.05):
    """Runs non-parametric and exact tests on the A/B test data."""
    print("\n[*] Running Statistical Tests...")
    
    old_data = df[df['Design'] == 'Old']
    new_data = df[df['Design'] == 'New']
    
    # 1. Satisfaction Analysis (Mann-Whitney U)
    stat, p_sat = stats.mannwhitneyu(old_data['Satisfaction'], new_data['Satisfaction'], alternative='two-sided')
    print("\n--- User Satisfaction (Mann-Whitney U) ---")
    print(f"P-value: {p_sat:.4f} -> {'Significant Difference' if p_sat < alpha else 'No Significant Difference'}")

    # 2. Time on Site Analysis (Mann-Whitney U)
    stat, p_time = stats.mannwhitneyu(old_data['TimeOnSite'], new_data['TimeOnSite'], alternative='two-sided')
    print("\n--- Time on Site (Mann-Whitney U) ---")
    print(f"P-value: {p_time:.4f} -> {'Significant Difference' if p_time < alpha else 'No Significant Difference'}")

    # 3. Purchase Completion (Fisher's Exact Test)
    contingency_table = pd.crosstab(df['Design'], df['Purchased'])
    odds_ratio, p_purchase = stats.fisher_exact(contingency_table)
    print("\n--- Purchase Completion (Fisher's Exact Test) ---")
    print(f"P-value: {p_purchase:.4f} -> {'Significant Difference' if p_purchase < alpha else 'No Significant Difference'}")

# 4. PREDICTIVE MODELING
def build_predictive_model(df):
    """Trains a Logistic Regression model to predict if a user will purchase based on design and behavior."""
    print("\n[*] Training Predictive Model (Logistic Regression)...")
    
    # Encode Design (Old = 0, New = 1)
    le = LabelEncoder()
    df['Design_Encoded'] = le.fit_transform(df['Design'])
    
    # Features & Target
    X = df[['Design_Encoded', 'Satisfaction', 'TimeOnSite']]
    y = df['Purchased']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("\nModel Evaluation (Predicting Purchase Completion):")
    print("-" * 45)
    print(classification_report(y_test, predictions, target_names=['Did Not Purchase (0)', 'Purchased (1)']))
    
    # Feature Importance (Coefficients)
    print("\nFeature Coefficients (Impact on Purchase):")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

# MAIN EXECUTION
if __name__ == "__main__":
    # 1. Load or Generate Data
    try:
        df = pd.read_csv('website_ab_test.csv')
        print("[*] Loaded existing A/B test dataset.")
    except FileNotFoundError:
        df = generate_ab_test_data()

    # 2. Execute Pipeline
    perform_eda(df)
    run_statistical_tests(df)
    build_predictive_model(df)