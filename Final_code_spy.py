import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'D:\My projects\Techpay.ai\Dataset\New folder\pc_market_data_synthetic.csv')
df.head()

df.describe()

# Convert 'Year' and 'Final Price' to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')

# Total Market Size Analysis
yearly_sales = df.groupby('Year')['Final Price'].sum()
growth_rate = yearly_sales.pct_change() * 100

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales.index, yearly_sales, marker='o', label='Yearly Sales')
plt.plot(growth_rate.index, growth_rate, marker='o', linestyle='--', label='Growth Rate')
plt.xlabel('Year')
plt.ylabel('Sales / Growth Rate (%)')
plt.title('Yearly Sales and Growth Rate')
plt.legend()
plt.grid(True)
plt.show()

# Price Band Analysis
def categorize_price(price):
    if 25000 <= price < 50000:
        return '₹25K - ₹50K'
    elif 50000 <= price < 75000:
        return '₹50K - ₹75K'
    elif 75000 <= price < 100000:
        return '₹75K - ₹100K'
    else:
        return '₹100K and above'

df['Price Band'] = df['Final Price'].apply(categorize_price)
price_band_sales = df.groupby('Price Band')['Final Price'].sum()
print(price_band_sales)

# Predicting Future Market Size
X = df[['Year']]
y = df['Final Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

future_years = pd.DataFrame({'Year': range(2024, 2029)})
future_predictions = model.predict(future_years)
print(future_predictions)
