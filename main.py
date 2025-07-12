import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure output directories exist
os.makedirs('outputs/visuals', exist_ok=True)

# Load and clean data
df = pd.read_csv('GenAI_Adoption_Impact.csv')
columns_needed = [
    'Company Name', 'Industry', 'Country', 'GenAI Tool', 'Adoption Year',
    'Number of Employees Impacted', 'New Roles Created', 'Training Hours Provided',
    'Productivity Change (%)', 'Employee Sentiment'
]
df = df[columns_needed].dropna()

# Visualization 1 - Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Productivity Change (%)'], kde=True)
plt.title('Distribution of Productivity Change After GenAI Adoption')
plt.xlabel('Productivity Change (%)')
plt.savefig('outputs/visuals/productivity_distribution.png')
plt.close()

# Visualization 2 - Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='GenAI Tool', y='Productivity Change (%)', data=df)
plt.xticks(rotation=45)
plt.title('Productivity Impact by GenAI Tool')
plt.savefig('outputs/visuals/productivity_by_tool.png')
plt.close()

# Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Employee Sentiment'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['Sentiment_Label'] = df['Sentiment_Score'].apply(
    lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
)

# Visualization 3 - Sentiment Summary
plt.figure(figsize=(6, 4))
df['Sentiment_Label'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'], title='Employee Sentiment Categories')
plt.ylabel('Count')
plt.savefig('outputs/visuals/sentiment_summary.png')
plt.close()

# Regression Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = df[['Number of Employees Impacted', 'New Roles Created', 'Training Hours Provided']]
y = df['Productivity Change (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Model Performance: RMSE = {rmse:.2f}, R² = {r2:.2f}")

# Clustering
from sklearn.cluster import KMeans

X_cluster = df[['Training Hours Provided', 'Number of Employees Impacted', 'Productivity Change (%)']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# Visualization 4 - Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Training Hours Provided',
    y='Productivity Change (%)',
    hue='Cluster',
    palette='Set2'
)
plt.title('Enterprise Clustering by Training & Productivity Change')
plt.savefig('outputs/visuals/clusters.png')
plt.close()

# Save model and data
import joblib
joblib.dump(model, 'outputs/model.pkl')
df.to_csv('outputs/final_dataset.csv', index=False)
print("Model and dataset saved.")