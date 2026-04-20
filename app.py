import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
sns.countplot(data=df, x='rating', palette='viridis')
plt.savefig('rating_distribution.png')
plt.clf()

cat_counts = df['category'].value_counts().reset_index()
cat_counts.columns = ['category', 'count']
plt.figure(figsize=(10, 5))
sns.barplot(data=cat_counts, x='count', y='category', palette='magma')
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
    acc = accuracy_score(y_test, y_pred)
    f1_neg = f1_score(y_test, y_pred, pos_label=0)
    bake_off_results.append({"Model": name, "Accuracy": acc, "F1_Negative": f1_neg})


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

if not os.path.exists('outputs'):
    os.makedirs('outputs')

# 1. Export Cleaned Data
df.to_csv('outputs/cleaned_reviews.csv', index=False)

# 2. Export Model Performance
pd.DataFrame(bake_off_results).to_csv('outputs/model_performance.csv', index=False)

# 3. Export Churn Drivers
pd.DataFrame(topic_results).to_csv('outputs/churn_drivers.csv', index=False)

# 4. Export Seller Rankings
seller_stats.sort_values(by='seller_score', ascending=False).to_csv('outputs/seller_rankings.csv', index=False)