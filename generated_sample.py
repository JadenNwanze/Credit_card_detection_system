import pandas as pd

# Load your full dataset
df = pd.read_csv("creditcard.csv")

# Take a sample of 2,500 legit and 2,500 fraud transactions
sample_df = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(n=2500, random_state=42))

# Save the sample to a smaller CSV file
sample_df.to_csv("sample_creditcard.csv", index=False)

print(" Sample CSV created: sample_creditcard.csv")
