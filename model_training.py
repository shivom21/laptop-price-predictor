import pandas as pd
import numpy as np
import pickle

# Load dataset
df = pd.read_excel("DATA2.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

df['Age'] = np.random.randint(0, 6, size=len(df))

final_df = df[['Company', 'TypeName', 'Ram', 'Memory', 'Age', 'Price']].copy()

def simplify_memory(mem):
    if 'SSD' in mem:
        return 1
    elif 'HDD' in mem:
        return 0
    else:
        return 2

final_df['Memory'] = final_df['Memory'].apply(simplify_memory)

df_model = pd.get_dummies(final_df, columns=['Company', 'TypeName'], drop_first=True)

X = df_model.drop('Price', axis=1)
y = df_model['Price']

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("Model trained and saved as model.pkl ✅")
