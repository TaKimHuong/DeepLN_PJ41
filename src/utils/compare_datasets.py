import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_paranoid(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Giữ lại cả chữ và số, bỏ hết dấu câu nhưng GIỮ KHOẢNG TRẮNG để so sánh cụm từ
    text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
    return " ".join(text.split())

def paranoid_compare():
    print("--- STARTING PARANOID AUDIT ---")
    
    # 1. Load Data
    df_old = pd.read_csv('data/raw/public_train.csv')
    new_file = r"H:\Downloads\ViFN-Vietnamese_Fake_New_Datasets_Ver3-main\ViFN-Vietnamese_Fake_New_Datasets_Ver3-main\processed\deduplicated_articles_fake.csv"
    df_new = pd.read_csv(new_file)
    
    content_cols = [c for c in ['title', 'summary'] if c in df_new.columns]
    
    # Chuẩn bị dữ liệu sạch
    print("Pre-processing datasets...")
    old_texts = df_old['post_message'].dropna().apply(clean_paranoid).tolist()
    
    new_data = []
    for idx, row in df_new.iterrows():
        full_text = " ".join([str(row[c]) for c in content_cols if pd.notna(row[c])])
        new_data.append(clean_paranoid(full_text))

    # 2. In mẫu để kiểm tra mắt (Visual Verification)
    print("\n[DEBUG] Sample Old Text (first 2):")
    for i in range(min(2, len(old_texts))):
        # Encode to ASCII for safe printing on Windows
        safe_txt = old_texts[i][:100].encode('ascii', 'ignore').decode()
        print(f"  Old {i}: {safe_txt}...")
        
    print("\n[DEBUG] Sample New Text (first 2):")
    for i in range(min(2, len(new_data))):
        safe_txt = new_data[i][:100].encode('ascii', 'ignore').decode()
        print(f"  New {i}: {safe_txt}...")


    # 3. Vòng 1: Substring Search (Kiểm tra xem tin mới có nằm trong tin cũ không)
    print("\nRunning Round 1: Substring checking...")
    substring_dups = 0
    unique_pass_1 = []
    
    # Chỉ check 100 tin đầu bằng substring vì nó rất chậm (O(N*M))
    # Hoặc dùng set hash cho nhanh trước
    old_hashes = set(old_texts)
    
    for i, txt in enumerate(new_data):
        if txt in old_hashes:
            substring_dups += 1
        else:
            # Check xem có là xâu con không (chỉ check với 1000 tin cũ đầu tiên để demo tốc độ)
            is_sub = False
            # for old in old_texts[:1000]: 
            #     if txt in old or old in txt:
            #         is_sub = True
            #         break
            if not is_sub:
                unique_pass_1.append(txt)
            else:
                substring_dups += 1
                
    print(f"  -> Found {substring_dups} duplicates via substring/exact match.")

    # 4. Vòng 2: TF-IDF với ngưỡng cực thấp (70%)
    print("\nRunning Round 2: Low-threshold Fuzzy Match (70%)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_old = vectorizer.fit_transform(old_texts)
    tfidf_new = vectorizer.transform(unique_pass_1)
    
    sim_matrix = cosine_similarity(tfidf_new, tfidf_old)
    max_sims = np.max(sim_matrix, axis=1)
    
    THRESHOLD = 0.70
    final_unique = []
    fuzzy_70_dups = 0
    
    for i, score in enumerate(max_sims):
        if score >= THRESHOLD:
            fuzzy_70_dups += 1
        else:
            final_unique.append(unique_pass_1[i])
            
    print(f"  -> Found {fuzzy_70_dups} records with >70% similarity.")

    print("\n" + "="*40)
    print("ABSOLUTE FINAL RESULTS:")
    print(f"Total New: {len(new_data)}")
    print(f"Duplicates: {substring_dups + fuzzy_70_dups}")
    print(f"STRICTLY UNIQUE: {len(final_unique)}")
    print("="*40)
    
    if final_unique:
        df_out = pd.DataFrame({'post_message': final_unique, 'label': 1})
        df_out.to_csv('data/raw/new_unique_fake_found.csv', index=False, encoding='utf-8')
        print(f"Done! Saved {len(final_unique)} unique records to data/raw/new_unique_fake_found.csv")


if __name__ == "__main__":
    paranoid_compare()
