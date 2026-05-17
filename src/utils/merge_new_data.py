import pandas as pd
import os
import datetime

def merge_data():
    print("--- Starting Data Merge ---")
    
    # 1. Load data cũ
    old_path = 'data/raw/public_train.csv'
    df_old = pd.read_csv(old_path)
    last_id = df_old['id'].max() if not df_old.empty else 0
    print(f"Old dataset size: {len(df_old)} rows. Last ID: {last_id}")
    
    # 2. Load 649 tin mới (tôi sẽ chạy lại logic lọc 70% để lấy đúng data)
    # Lưu ý: Ở bước trước tôi đã lưu vào file tạm, giờ tôi sẽ dùng file đó
    # Nhưng để chắc chắn nhất, tôi sẽ load trực tiếp từ bộ so khớp 70%
    unique_file = 'data/raw/new_unique_fake_found.csv'
    if not os.path.exists(unique_file):
        print("Error: Unique fake news file not found. Please run comparison first.")
        return
        
    df_new_raw = pd.read_csv(unique_file)
    print(f"Loading {len(df_new_raw)} unique new fake news records...")

    # 3. Chuyển đổi sang định dạng của public_train.csv
    new_rows = []
    current_id = last_id + 1
    
    # Lấy timestamp hiện tại
    now_ts = int(datetime.datetime.now().timestamp())
    
    for idx, row in df_new_raw.iterrows():
        new_rows.append({
            'id': current_id,
            'user_name': 'vifn_v3_dataset',
            'post_message': row['post_message'],
            'timestamp_post': now_ts,
            'num_like_post': 0,
            'num_comment_post': 0,
            'num_share_post': 0,
            'label': 1
        })
        current_id += 1
        
    df_to_append = pd.DataFrame(new_rows)
    
    # 4. Gộp
    df_final = pd.concat([df_old, df_to_append], ignore_index=True)
    
    # 5. Lưu lại
    df_final.to_csv(old_path, index=False, encoding='utf-8')
    
    print("\nMerge Results:")
    print(f"- Total rows added: {len(df_to_append)}")
    print(f"- Final dataset size: {len(df_final)} rows")
    print(f"- Data saved to: {old_path}")

if __name__ == "__main__":
    merge_data()
