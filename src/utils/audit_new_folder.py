import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def clean_paranoid(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
    return " ".join(text.split())

def audit_with_samples():
    print("--- STARTING DETAILED DUPLICATE AUDIT ---")
    
    # Load OLD Data from data_03
    old_path = 'data_03/raw/public_train.csv'
    df_old = pd.read_csv(old_path)
    old_texts = df_old['post_message'].dropna().apply(clean_paranoid).tolist()
    old_hashes = {t: i for i, t in enumerate(old_texts)} # Lưu hash kèm index để tra cứu
    
    # Collect NEW Fake News
    new_fake_raw = []
    base_dir = r"H:\Downloads\Tiếng Việt"
    
    # Gom dữ liệu (chỉ lấy tin giả)
    for f in ["vn_news_223_tdlfr.csv", "vn_news_226_tlfr.csv"]:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            df = pd.read_csv(p)
            for t in df[df['label'] == 1]['text'].tolist():
                new_fake_raw.append(t)
                
    pq_train = os.path.join(base_dir, "train-00000-of-00001.parquet")
    if os.path.exists(pq_train):
        df = pd.read_parquet(pq_train)
        for t in df[df['labels'] == 1]['Statement'].tolist():
            new_fake_raw.append(t)

    print(f"Collected {len(new_fake_raw)} new fake news candidates.")
    
    exact_duplicates = []
    unique_pass_1 = []
    
    print("Checking for Exact Duplicates...")
    for txt in new_fake_raw:
        c_txt = clean_paranoid(txt)
        if len(c_txt) < 30: continue
        
        if c_txt in old_hashes:
            old_idx = old_hashes[c_txt]
            exact_duplicates.append((txt, df_old.iloc[old_idx]['post_message']))
        else:
            unique_pass_1.append(txt)
            
    print(f"Found {len(exact_duplicates)} exact duplicates.")
    
    # In ra 3 mẫu tin trùng để kiểm chứng
    if exact_duplicates:
        print("\n[EXACT DUPLICATE SAMPLES]:")
        for i in range(min(3, len(exact_duplicates))):
            new_t, old_t = exact_duplicates[i]
            print(f"\n--- Sample {i+1} ---")
            print(f"NEW: {new_t[:150].encode('ascii', 'ignore').decode()}...")
            print(f"OLD: {old_t[:150].encode('ascii', 'ignore').decode()}...")


    # Fuzzy check
    print("\nChecking for Fuzzy Duplicates (>80%)...")
    if unique_pass_1:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_old = vectorizer.fit_transform(old_texts)
        tfidf_new = vectorizer.transform([clean_paranoid(t) for t in unique_pass_1])
        
        sim_matrix = cosine_similarity(tfidf_new, tfidf_old)
        max_sims = np.max(sim_matrix, axis=1)
        
        fuzzy_list = []
        final_unique = []
        for i, score in enumerate(max_sims):
            if score > 0.80:
                old_idx = np.argmax(sim_matrix[i])
                fuzzy_list.append((unique_pass_1[i], df_old.iloc[old_idx]['post_message'], score))
            else:
                final_unique.append(unique_pass_1[i])
                
        print(f"Found {len(fuzzy_list)} fuzzy duplicates (>80%).")
        if fuzzy_list:
            print("\n[FUZZY DUPLICATE SAMPLES (>80%)]:")
            for i in range(min(2, len(fuzzy_list))):
                new_t, old_t, score = fuzzy_list[i]
                print(f"\n--- Sample {i+1} (Similarity: {score:.2f}) ---")
                print(f"NEW: {new_t[:150].encode('ascii', 'ignore').decode()}...")
                print(f"OLD: {old_t[:150].encode('ascii', 'ignore').decode()}...")


    print("\n" + "="*40)
    print(f"KẾT QUẢ CUỐI CÙNG:")
    print(f"- Trùng tuyệt đối: {len(exact_duplicates)}")
    print(f"- Trùng tương đối: {len(fuzzy_list) if 'fuzzy_list' in locals() else 0}")
    print(f"- TIN MỚI THỰC SỰ: {len(final_unique) if 'final_unique' in locals() else len(unique_pass_1)}")
    print("="*40)

if __name__ == "__main__":
    audit_with_samples()
