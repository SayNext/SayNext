import pandas as pd
import sys

def xlsx_to_csv(xlsx_path, csv_path):
    # 读取 Excel 文件
    df = pd.read_excel(xlsx_path, dtype=str)  # 强制所有列读取为字符串
    # 保存为 CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python xlsx_to_csv.py 输入文件.xlsx 输出文件.csv")
    else:
        xlsx_to_csv(sys.argv[1], sys.argv[2])
        print(f"转换完成: {sys.argv[2]}")
