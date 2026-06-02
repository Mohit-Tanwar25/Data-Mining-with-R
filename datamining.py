# Apriori Algorithm - Menu Driven Program with Excel Input
# Requires: pandas, mlxtend, openpyxl

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------- getch() ----------
def getch():
    input("\nPress Enter to continue...")

# ---------- Global Variables ----------
transactions = []
df = None
frequent_itemsets = None

# ---------- Functions ----------

def load_excel():
    global transactions
    
    file_path = "mohit.xlsx"
    # file_path = input("Enter Excel file path (e.g., data.xlsx): ")
    
    try:
        # Read Excel file
        data = pd.read_excel(file_path, engine='openpyxl')
        
        print("\nExcel Data Loaded Successfully!")
        print(data.head())
        
        # Convert rows into transaction list
        transactions.clear()
        
        for index, row in data.iterrows():
            # Drop NaN and convert to list
            items = [str(item) for item in row if pd.notna(item)]
            transactions.append(items)
        
        print("\nTransactions Created Successfully!")
        
    except Exception as e:
        print("Error loading file:", e)

def show_transactions():
    if not transactions:
        print("\n⚠ No data loaded!")
        return
    
    print("\n--- Transactions ---")
    for i, t in enumerate(transactions, start=1):
        print(f"T{i}: {t}")

def convert_to_dataframe():
    global df
    
    if not transactions:
        print("\n⚠ Load Excel data first!")
        return
    
    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    
    df = pd.DataFrame(te_data, columns=te.columns_)
    
    print("\n--- One-Hot Encoded Data ---")
    print(df)

def run_apriori():
    global df, frequent_itemsets
    
    if df is None:
        print("\n⚠ Convert data first!")
        return
    
    min_sup = float(input("Enter minimum support (e.g., 0.3): "))
    
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True)
    
    print("\n--- Frequent Itemsets ---")
    print(frequent_itemsets)

def generate_rules():
    global frequent_itemsets
    
    if frequent_itemsets is None:
        print("\n⚠ Run Apriori first!")
        return
    
    min_conf = float(input("Enter minimum confidence (e.g., 0.6): "))
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    
    if rules.empty:
        print("\nNo rules found!")
    else:
        print("\n--- Association Rules ---")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ---------- Menu ----------
while True:
    print("\n==============================")
    print(" APRIORI ALGORITHM (EXCEL)")
    print("==============================")
    print("1. Load Data from Excel")
    print("2. Show Transactions")
    print("3. Convert to One-Hot Data")
    print("4. Run Apriori Algorithm")
    print("5. Generate Association Rules")
    print("6. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        load_excel()
        getch()
    
    elif choice == '2':
        show_transactions()
        getch()
    
    elif choice == '3':
        convert_to_dataframe()
        getch()
    
    elif choice == '4':
        run_apriori()
        getch()
    
    elif choice == '5':
        generate_rules()
        getch()
    
    elif choice == '6':
        print("Exiting program...")
        break
    
    else:
        print("Invalid choice!")
        getch()