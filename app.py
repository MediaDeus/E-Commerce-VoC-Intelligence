import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# --- DATA CLEANING ---
# ==========================================
print("--- DATA CLEANING ---")
df = pd.read_csv('tokopedia-product-reviews-2019.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df['sold'] = df['sold'].fillna('0')
df['sold'] = df['sold'].astype(str).apply(lambda x: re.sub(r'\D', '', x))
df['sold'] = df['sold'].apply(lambda x: int(x) if x != '' else 0)

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text
    
df['cleaned_text'] = df['text'].apply(clean_text)
df = df.dropna(subset=['text', 'rating'])
df = df[df['cleaned_text'].str.len() > 0]


# ==========================================
# --- EXPLORATORY DATA ANALYSIS ---
# ==========================================
print("\n--- EXPLORATORY DATA ANALYSIS (EDA) ---")
df['review_length'] = df['text'].astype(str).apply(len)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rating', palette='viridis', hue='rating', legend=False)
plt.savefig('rating_distribution.png')
plt.clf()

cat_counts = df['category'].value_counts().reset_index()
cat_counts.columns = ['category', 'count']
plt.figure(figsize=(10, 5))
sns.barplot(data=cat_counts, x='count', y='category', palette='magma', hue='category', legend=False)
plt.savefig('category_distribution.png')
plt.clf()


# ==========================================
# --- BINARY SENTIMENT BAKE-OFF ---
# ==========================================
print("\n--- BINARY SENTIMENT BAKE-OFF ---")
df_binary = df[df['rating'] != 3].copy()
df_binary['sentiment'] = df_binary['rating'].apply(lambda x: 1 if x >= 4 else 0)

X = df_binary['cleaned_text']
y = df_binary['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) 
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Naive_Bayes": MultinomialNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
}

bake_off_results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Get overall accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Generate the classification report as a dictionary so we can extract exact numbers
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Append the highly detailed metrics to our results list
    bake_off_results.append({
        "Model": name, 
        "Overall_Accuracy": acc, 
        "Negative_Precision": report['0']['precision'],
        "Negative_Recall": report['0']['recall'],
        "Negative_F1": report['0']['f1-score'],
        "Positive_Precision": report['1']['precision'],
        "Positive_Recall": report['1']['recall'],
        "Positive_F1": report['1']['f1-score']
    })

# ==========================================
# --- PHASE 2.5: DEEP DIVE MODEL EVALUATION ---
# ==========================================
print("\n--- STARTING PHASE 2.5: DETAILED MODEL EVALUATION ---")
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# 1. Isolate the winning model
best_model = models["Logistic_Regression"] 

# Get standard predictions and probability scores
y_pred_best = best_model.predict(X_test_tfidf)
y_prob_best = best_model.predict_proba(X_test_tfidf)[:, 1]

# --- CHART A: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative (0)', 'Positive (1)'], 
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.ylabel('Actual Sentiment (Truth)')
plt.xlabel('Predicted Sentiment (AI Guess)')
plt.title('Confusion Matrix: Where does the AI make mistakes?')
plt.tight_layout()
plt.savefig('outputs/model_confusion_matrix.png')
plt.clf()

# --- CHART B: ROC Curve & AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Model Confidence')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('outputs/model_roc_curve.png')
plt.clf()

# --- CHART C: Feature Importance (Explainable AI) ---
feature_names_tfidf = tfidf.get_feature_names_out()
coefficients = best_model.coef_[0]

importance_df = pd.DataFrame({'Word': feature_names_tfidf, 'Weight': coefficients})

top_positive = importance_df.sort_values(by='Weight', ascending=False).head(10)
top_negative = importance_df.sort_values(by='Weight', ascending=True).head(10)

top_features = pd.concat([top_positive, top_negative]).sort_values(by='Weight')

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features['Weight']]
sns.barplot(x='Weight', y='Word', data=top_features, palette=colors, hue='Word', legend=False)
plt.title('Model Explainability: Words Driving Sentiment Predictions')
plt.xlabel('Mathematical Weight (Negative = Churn Risk, Positive = Happy)')
plt.ylabel('TF-IDF Keyword')
plt.tight_layout()
plt.savefig('outputs/model_feature_importance.png')
plt.clf()


# ==========================================
# --- TOPIC MODELING ---
# ==========================================
print("\n--- TOPIC MODELING ---")
negative_reviews = df[df['rating'] <= 2]['cleaned_text']
custom_stop_words = ['yang', 'di', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'tidak', 'ke', 'dari', 'ada', 'yg', 'barang', 'nya', 'saya', 'tapi']
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words)
X_neg = vectorizer.fit_transform(negative_reviews)

lda_model = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=10)
lda_model.fit(X_neg)
feature_names = vectorizer.get_feature_names_out()

topic_results = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    topic_results.append({"Topic": topic_idx + 1, "Keywords": ", ".join(top_words)})


# ==========================================
# --- SELLER QUALITY RANKING ---
# ==========================================
print("\n--- SELLER QUALITY RANKING ---")
seller_stats = df.groupby('shop_id').agg({'rating': 'mean', 'sold': 'sum', 'text': 'count'}).reset_index()
seller_stats = seller_stats[seller_stats['text'] >= 5]
scaler = MinMaxScaler()
seller_stats[['norm_rating', 'norm_sold']] = scaler.fit_transform(seller_stats[['rating', 'sold']])
seller_stats['seller_score'] = (seller_stats['norm_rating'] * 0.7) + (seller_stats['norm_sold'] * 0.3)

# --- EXPORT ---

# 1. Export Cleaned Data
df.to_csv('outputs/cleaned_reviews.csv', index=False)

# 2. Export Model Performance
# 2. Export Detailed Model Performance
pd.DataFrame(bake_off_results).to_csv('outputs/detailed_model_performance.csv', index=False)

# 3. Export Churn Drivers
pd.DataFrame(topic_results).to_csv('outputs/churn_drivers.csv', index=False)

# 4. Export Seller Rankings
seller_stats.sort_values(by='seller_score', ascending=False).to_csv('outputs/seller_rankings.csv', index=False)


# ==========================================
# --- PHASE 6: EXHAUSTIVE VISUALIZATION EXPORT ---
# ==========================================
print("\n--- STARTING PHASE 6: EXPORTING ARTIFACTS & VISUAL REPORTS ---")

# --- 1. Export Raw CSV Artifacts ---
df.to_csv('outputs/cleaned_reviews.csv', index=False)
seller_stats.sort_values(by='seller_score', ascending=False).to_csv('outputs/seller_rankings.csv', index=False)

print("Generating comprehensive business dashboards (PNGs)...")
sns.set_theme(style="whitegrid")

# Helper function to save plots cleanly
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}')
    plt.clf()

# --- CHART 1 & 2: Top & Worst Products (Things) ---
item_col = 'product_name' if 'product_name' in df.columns else 'shop_id' 
item_sales = df.groupby(item_col)['sold'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
top_items = item_sales.head(10)
sns.barplot(x=top_items.values, y=top_items.index.astype(str), hue=top_items.index.astype(str), palette='Blues_r', legend=False)
plt.title('1. Top 10 Best-Selling Items')
plt.xlabel('Total Units Sold')
save_plot('1_top_sales_things.png')

plt.figure(figsize=(10, 6))
worst_items = item_sales[item_sales > 0].tail(10) 
sns.barplot(x=worst_items.values, y=worst_items.index.astype(str), hue=worst_items.index.astype(str), palette='Reds_r', legend=False)
plt.title('2. Bottom 10 Worst-Selling Items (Non-Zero)')
plt.xlabel('Total Units Sold')
save_plot('2_worst_sales_things.png')


# --- CHART 3 & 4: Top & Worst Sellers ---
seller_sales = df.groupby('shop_id')['sold'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
top_sellers_vol = seller_sales.head(10)
sns.barplot(x=top_sellers_vol.values, y=top_sellers_vol.index.astype(str), hue=top_sellers_vol.index.astype(str), palette='Greens_r', legend=False)
plt.title('3. Top 10 Sellers by Total Sales Volume')
plt.xlabel('Total Units Sold')
save_plot('3_top_sellers.png')

plt.figure(figsize=(10, 6))
worst_sellers_vol = seller_sales[seller_sales > 0].tail(10)
sns.barplot(x=worst_sellers_vol.values, y=worst_sellers_vol.index.astype(str), hue=worst_sellers_vol.index.astype(str), palette='Oranges_r', legend=False)
plt.title('4. Bottom 10 Sellers by Total Sales Volume (Non-Zero)')
plt.xlabel('Total Units Sold')
save_plot('4_worst_sellers.png')

# --- CHART 7: Most Used Words in All Reviews ---
print("Calculating overall word frequencies...")
custom_stop_words = ['yang', 'di', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'tidak', 'ke', 'dari', 'ada', 'yg', 'barang', 'nya', 'saya', 'tapi']
vec_all = CountVectorizer(max_features=20, stop_words=custom_stop_words)
X_all = vec_all.fit_transform(df['cleaned_text'])
word_freq = pd.DataFrame({'word': vec_all.get_feature_names_out(), 'freq': X_all.sum(axis=0).A1}).sort_values(by='freq', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='freq', y='word', hue='word', data=word_freq, palette='magma', legend=False)
plt.title('7. Top 20 Most Used Words in All Reviews')
plt.xlabel('Frequency')
save_plot('7_most_used_words.png')


# --- CHART 8: Sentiment Distribution ---
plt.figure(figsize=(6, 6))
plt.pie(df_binary['sentiment'].value_counts(), labels=['Positive (4-5 Stars)', 'Negative (1-2 Stars)'], 
        autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.1))
plt.title('8. Platform Sentiment Distribution')
save_plot('8_sentiment_distribution.png')


# --- CHART 9 (DATA SCIENCE BONUS): Sentiment Health by Category ---
plt.figure(figsize=(12, 6))
sentiment_cat = df_binary.groupby('category')['sentiment'].mean().sort_values() * 100
sns.barplot(x=sentiment_cat.values, y=sentiment_cat.index.astype(str), hue=sentiment_cat.index.astype(str), palette='RdYlGn', legend=False)
plt.title('9. Sentiment Health: % of Positive Reviews by Category')
plt.xlabel('Percentage of Positive Reviews (%)')
plt.xlim(0, 100)
save_plot('9_sentiment_by_category.png')

print("Success! All visualizations have been exported to the 'outputs' folder.")