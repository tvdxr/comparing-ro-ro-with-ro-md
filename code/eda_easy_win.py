import os
import json
from pathlib import Path
from collections import Counter
import statistics
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('stopwords', quiet=True)  # quiet=True suppresses output
from nltk.corpus import stopwords

print("="*80)
print("EXPLORATORY DATA ANALYSIS - RORO-ANALIZA DATA-CLEANED FOLDER")
print("="*80)

# Initialize data structures
data_folder = Path('data-cleaned')
all_files = []  # Store all JSON articles
categories = Counter()  # Count articles per category
regions = Counter()  # Count articles per region

print("\n1. Loading data from data-cleaned folder...")
print("-" * 80)

# Load all JSON files from data-cleaned folder
for json_file in data_folder.rglob('*.json'):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            article = json.load(f)
            article['file_path'] = str(json_file)
            all_files.append(article)
            
            # Extract category and region from file path
            parts = json_file.parts
            if len(parts) >= 3:
                category = parts[1]  # 'judete', 'raioane', 'int', 'int_istoric'
                region = parts[2]    # specific region name
                categories[category] += 1
                regions[region] += 1
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

print(f"✓ Total files loaded: {len(all_files)}")

print(f"\nCategories found:")
for cat, count in categories.most_common():
    print(f"  - {cat}: {count}")

print(f"\nTop 15 regions found:")
for region, count in regions.most_common(15):
    print(f"  - {region}: {count}")

print("\n" + "="*80)
print("2. TEXT CONTENT ANALYSIS")
print("="*80)

# Calculate text metrics for all articles
title_lengths = []
content_lengths = []
title_word_counts = []
content_word_counts = []

for article in all_files:
    title = article.get('title', '')
    content = article.get('content', '')
    
    # Store length and word count for each article
    title_lengths.append(len(title))
    content_lengths.append(len(content))
    title_word_counts.append(len(title.split()))
    content_word_counts.append(len(content.split()))

print(f"\nTitle Statistics:")
print(f"  Average length: {statistics.mean(title_lengths):.0f} characters")
print(f"  Median length: {statistics.median(title_lengths):.0f} characters")
print(f"  Min length: {min(title_lengths)} characters")
print(f"  Max length: {max(title_lengths)} characters")

print(f"\nContent Statistics:")
content_nonzero = [x for x in content_lengths if x > 0]
print(f"  Average length: {statistics.mean(content_lengths):.0f} characters")
print(f"  Median length: {statistics.median(content_lengths):.0f} characters")
print(f"  Min length: {min(content_nonzero)} characters")
print(f"  Max length: {max(content_lengths)} characters")

print(f"\nWord Count Statistics:")
print(f"  Average title words: {statistics.mean(title_word_counts):.0f} words")
print(f"  Average content words: {statistics.mean(content_word_counts):.0f} words")

print("\n" + "="*80)
print("3. DATA COMPLETENESS")
print("="*80)

# Check how many articles have each field populated
titles_present = sum(1 for a in all_files if a.get('title'))
content_present = sum(1 for a in all_files if a.get('content'))
metadata_present = sum(1 for a in all_files if a.get('metadata'))

print(f"  Titles present: {titles_present}/{len(all_files)} ({titles_present/len(all_files)*100:.1f}%)")
print(f"  Content present: {content_present}/{len(all_files)} ({content_present/len(all_files)*100:.1f}%)")
print(f"  Metadata present: {metadata_present}/{len(all_files)} ({metadata_present/len(all_files)*100:.1f}%)")

print("\n" + "="*80)
print("4. METADATA ANALYSIS")
print("="*80)

file_extensions = Counter()
for article in all_files:
    meta = article.get('metadata', {})
    if isinstance(meta, dict) and 'original_file' in meta:
        original_file = meta['original_file']
        if '.' in original_file:
            # file extension (.html)
            ext = original_file.split('.')[-1]
            file_extensions[ext] += 1

print(f"Original file extensions found:")
for ext, count in file_extensions.most_common():
    print(f"  .{ext}: {count}")

print("\n" + "="*80)
print("5. LANGUAGE COVERAGE")
print("="*80)

ro_ro_count = sum(1 for a in all_files if 'raioane' not in a['file_path'])
ro_md_count = sum(1 for a in all_files if 'raioane' in a['file_path'])

print(f"  ✓ Romanian (Romania) - ro-RO: {ro_ro_count:,} articles")
print(f"    - Categories: 'judete' (16,983), 'int' (2,422), 'int_istoric' (4,144)")
print(f"  ✓ Romanian (Moldova) - ro-MD: {ro_md_count:,} articles")
print(f"    - Category: 'raioane' (from RepMoldova region)")

print("\n" + "="*80)
print("6. STATISTICS BY CATEGORY")
print("="*80)

category_stats = {}
for category in categories.keys():
    cat_articles = [a for a in all_files if category in a['file_path']]
    if cat_articles:
        cat_content_lengths = [len(a.get('content', '')) for a in cat_articles]
        cat_content_words = [len(a.get('content', '').split()) for a in cat_articles]
        
        category_stats[category] = {
            'count': len(cat_articles),
            'avg_length': statistics.mean(cat_content_lengths),
            'avg_words': statistics.mean(cat_content_words),
        }

for category in sorted(category_stats.keys()):
    stats = category_stats[category]
    print(f"\n{category}:")
    print(f"  Articles: {stats['count']}")
    print(f"  Avg length: {stats['avg_length']:.0f} characters")
    print(f"  Avg words: {stats['avg_words']:.0f} words")

# word statistics 
print("\n" + "="*80)
print("7. WORD STATISTICS")
print("="*80)

# word len > 3
def analyze_words(articles_list):
    word_counter = Counter()
    romanian_stops = set(stopwords.words('romanian'))
    for article in articles_list:
        content = article.get('content', '')
        title = article.get('title', '')
        words = content.split() + title.split()
        for word in words:
            clean_word = word.strip('.,!?;"()[]').lower()
            if clean_word not in romanian_stops and len(clean_word) > 3:
                word_counter[clean_word] += 1
                
    df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['count'])
    
    if df.empty:
        return df
    
    df['percentage'] = (df['count'] /df['count'].sum()) * 100
    
    top5 = df.sort_values(by='percentage', ascending=False).head(5)
    
    return top5

# word len <= 3
def analyze_small_diff(articles_list):
    word_counter = Counter()
    romanian_stops = set(stopwords.words('romanian'))
    not_wrds = {" ", "-", "", "–", "•"}
    
    for article in articles_list:
        content = article.get('content', '')
        title = article.get('title', '')
        words = content.split() + title.split()
        for word in words:
            clean_word = word.strip('.,!?;"()[]{}').lower()
            if (clean_word and 
                clean_word not in romanian_stops and 
                len(clean_word) <= 3 and 
                clean_word not in not_wrds):
                word_counter[clean_word] += 1
                
    df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['count'])
    
    if df.empty:
        return df
    
    df['percentage'] = (df['count'] /df['count'].sum()) * 100
    
    top5 = df.sort_values(by='percentage', ascending=False).head(5)
    
    return top5

print(analyze_small_diff(all_files))
print(analyze_words(all_files))

ro_corpus = [a for a in all_files if 'raioane' not in a['file_path']]
md_corpus = [a for a in all_files if 'raioane'  in a['file_path']]

def lang_diff(ro_corpus, md_corpus): 
    # ro-ro = 0; ro-md = 1
    texts = [a.get('content', '') for a in ro_corpus] + [a.get('content', '') for a in md_corpus]
    labels = [0] * len(ro_corpus) + [1] * len(md_corpus)

    # vectorization - ngram_range=(2, 5) looks for 2-word and 5-word phrases
    cv = CountVectorizer(
        # linking words removal (optional)
        # stop_words = list(stopwords.words('romanian')),
        ngram_range = (2, 5),
        min_df = 5,
        lowercase = True
    )
    X = cv.fit_transform(texts)

    # model training c = 0.1 for strongest features
    # model uses as few words as possible to separate the two classes
    model = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
    model.fit(X, labels)
    
    # feature extraction
    feature_names = cv.get_feature_names_out()
    coefs = model.coef_[0] # weight of every phrase
    
    # results 
    df_divergence = pd.DataFrame({'phrase' : feature_names, 'importance': coefs})
    df_divergence = df_divergence.sort_values(by='importance', ascending=False)
    
    return df_divergence

def print_divergent_phrases(df_divergence, top_n=15):
    """Print top Moldova-specific and Romania-specific phrases."""
    print("\n" + "="*80)
    print("DIVERGENT PHRASES ANALYSIS")
    print("="*80)
    
    print(f"\n Top {top_n} Moldova-specific phrases (RO-MD):")
    print(df_divergence.head(top_n).to_string())
    
    print(f"\n Top {top_n} Romania-specific phrases (RO-RO):")
    print(df_divergence.tail(top_n).to_string())

df_divergence = lang_diff(ro_corpus, md_corpus)
print_divergent_phrases(df_divergence, top_n=15)

print("\n" + "="*80)
print("8. FINAL SUMMARY")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total articles: {len(all_files):,}")
print(f"  Total unique regions: {len(regions)}")
print(f"  Total categories: {len(categories)}")

print(f"\nContent Characteristics:")
print(f"  Average article: {statistics.mean(content_lengths):.0f} characters (~{statistics.mean(content_word_counts):.0f} words)")
print(f"  Longest article: {max(content_lengths):,} characters")
print(f"  Articles present: {content_present}/{len(all_files)} ({content_present/len(all_files)*100:.1f}%)")

print(f"\nLanguage Mix:")
print(f"  ✓ Romanian (Romania) - ro-RO: {ro_ro_count:,} articles ({ro_ro_count/len(all_files)*100:.1f}%)")
print(f"  ✓ Romanian (Moldova) - ro-MD: {ro_md_count:,} articles ({ro_md_count/len(all_files)*100:.1f}%)")

print("\n✓ Analysis complete!")