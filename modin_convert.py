import modin.pandas as pd

csv_filename = 'rights.csv'
txt_filename = 'gen.txt'

# Read CSV data into a Modin DataFrame
df = pd.read_csv(csv_filename)

# Open and write to the text file
with open(txt_filename, 'w', encoding='utf-8') as txtfile:
    for _, row in df.iterrows():
        question, answer = row['Question'], row['Answer']
        txtfile.write(f"Question: {question}\nAnswer: {answer}\n")

print(f"Conversion completed. Data written to {txt_filename}")
