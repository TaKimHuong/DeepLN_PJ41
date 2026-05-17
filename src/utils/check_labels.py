import pandas as pd

def check_labels():
    path = r"H:\Downloads\Tiếng Việt\train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    
    for lbl in [0, 1, 2]:
        print(f"\n--- LABEL {lbl} ---")
        samples = df[df['labels'] == lbl]['Statement'].head(2).tolist()
        for s in samples:
            # Bỏ dấu để in ra console Windows không bị lỗi
            safe_s = s.encode('ascii', 'ignore').decode()
            print(f"- {safe_s[:200]}...")

if __name__ == "__main__":
    check_labels()
