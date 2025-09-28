import pandas as pd

file1 = "/home/unsettledaverage73/ABB-chatbot/abb-chatbot/3439_RMU_StockObso_2507A-1(1).csv"
file2 = "/home/unsettledaverage73/ABB-chatbot/abb-chatbot/sustainbility2(1).csv"

print(f"\n--- Analyzing {file1} ---")
try:
    df1 = pd.read_excel(file1)
    print("Head of the dataframe:")
    print(df1.head())
    print("\nColumns:")
    print(df1.columns)
    print("\nSummary statistics:")
    print(df1.describe())
except Exception as e:
    print(f"Error reading {file1}: {e}")

print(f"\n--- Analyzing {file2} ---")
try:
    df2 = pd.read_excel(file2)
    print("Head of the dataframe:")
    print(df2.head())
    print("\nColumns:")
    print(df2.columns)
    print("\nSummary statistics:")
    print(df2.describe())
except Exception as e:
    print(f"Error reading {file2}: {e}")
