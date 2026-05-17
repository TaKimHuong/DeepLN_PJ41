import pandas as pd
import re
import os

def clean_paranoid(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ\s]', ' ', text)
    return " ".join(text.split())

def audit_label_0():
    print("--- STARTING LABEL 0 SUPREME AUDIT ---")
    
    # 1. Load Baseline data_03
    df_old = pd.read_csv('data_03/raw/public_train.csv')
    old_hashes = set(df_old['post_message'].dropna().apply(clean_paranoid))
    
    # 2. Collect Label 0 from new CSVs
    base_dir = r"H:\Downloads\Tiếng Việt"
    new_label_0_raw = []
    
    for f in ["vn_news_223_tdlfr.csv", "vn_news_226_tlfr.csv"]:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            df = pd.read_csv(p)
            # CHỈ LẤY LABEL 0
            new_label_0_raw.extend(df[df['label'] == 0]['text'].tolist())

    print(f"Total Label 0 candidates collected: {len(new_label_0_raw)}")
    
    # 3. In mẫu 10 tin để bạn kiểm tra độ sạch
    print("\n[CHECKING 5 SAMPLES FROM LABEL 0]:")

    for i in range(min(5, len(new_label_0_raw))):
        safe_txt = new_label_0_raw[i][:150].encode('ascii', 'ignore').decode()
        print(f"Sample {i+1}: {safe_txt}...")

    # 4. Kiểm tra trùng lặp
    exact_dups = 0
    unique_label_0 = []
    
    for txt in new_label_0_raw:
        c_txt = clean_paranoid(txt)
        if c_txt in old_hashes:
            exact_dups += 1
        else:
            unique_label_0.append(txt)
            
    print(f"\n[COMPARISON RESULTS]:")
    print(f"- Total Label 0 messages: {len(new_label_0_raw)}")
    print(f"- DUPLICATES found in data_03: {exact_dups}")
    print(f"- UNIQUE NEW fake news: {len(unique_label_0)}")


if __name__ == "__main__":
    audit_label_0()
