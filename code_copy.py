import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'D:\My projects\Techpay.ai\Dataset\New folder\pc_market_data_synthetic.csv')
df.head()

df.describe()

df['Final Price'] = pd.to_numeric(df['Final Price'], errors='coerce')
yearly_sales = df.groupby('Year')['Final Price'].sum()
print(yearly_sales)

growth_rate = yearly_sales.pct_change() * 100
print(growth_rate)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(yearly_sales.index, yearly_sales, marker='o', label='Yearly Sales')
plt.plot(growth_rate.index, growth_rate, marker='o', linestyle='--', label='Growth Rate')
plt.xlabel('Year')
plt.ylabel('Sales / Growth Rate (%)')
plt.title('Yearly Sales and Growth Rate')
plt.legend()
plt.grid(True)
plt.show()


region_sales = df.groupby('Region')['Final Price'].sum()
print(region_sales)

state_sales = df.groupby('State')['Final Price'].sum()
city_sales = df.groupby('City')['Final Price'].sum()
print(state_sales)
print(city_sales)

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

gender_sales = df.groupby('Gender')['Final Price'].sum()
print(gender_sales)


top_cities = df['City'].value_counts().head(5).index
top_city_sales = df[df['City'].isin(top_cities)].groupby('City')['Final Price'].sum()
print(top_city_sales)

tier_2_3_cities = df[~df['City'].isin(top_cities)]
tier_2_3_city_sales = tier_2_3_cities.groupby('City')['Final Price'].sum()
print(tier_2_3_city_sales)


retailer_sales = df.groupby('Retailer Type')['Final Price'].sum()
print(retailer_sales)



# Drop rows with NaN values in 'Retailer Type'
df = df.dropna(subset=['Retailer Type'])

# Calculate online and offline sales
online_sales = df[df['Retailer Type'].str.contains('Online')]['Final Price'].sum()
offline_sales = df[~df['Retailer Type'].str.contains('Online')]['Final Price'].sum()

print(f"Online Sales: {online_sales}")
print(f"Offline Sales: {offline_sales}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Prepare features and target variable
df['Year'] = df['Year'].astype(int)
# Predicting Future Market Size
X = df[['Year']]
y = df['Final Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

future_years = pd.DataFrame({'Year': range(2024, 2029)})
future_predictions = model.predict(future_years)
print(future_predictions)


###################################################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Prepare the data
X = df[['Year']]
y = df['Final Price']

# Add polynomial features
poly = PolynomialFeatures(degree=2)  # You can experiment with different degrees
X_poly = poly.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future years with polynomial features
future_years = pd.DataFrame({'Year': range(2024, 2029)})
future_years_poly = poly.transform(future_years)
future_predictions = model.predict(future_years_poly)

# Create a DataFrame for the output
output = pd.DataFrame({
    'Year': range(2024, 2029),
    'Predicted Final Price': future_predictions
})

# Print the output in the desired format
for index, row in output.iterrows():
    print(f"{int(row['Year'])}-{row['Predicted Final Price']:.2f}")

