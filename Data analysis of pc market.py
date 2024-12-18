import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'D:\My projects\Techpay.ai\Dataset\New folder\pc_market_data_synthetic.csv')
df.head()

df.describe()



df.info()
df.isnull().sum()
# Remove rows where 'GPU' is missing
df = df.dropna(subset=['GPU'])

#Visualize Data Distribution
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# RAM Distribution
# RAM Distribution
sns.histplot(df["RAM"], kde=True)
plt.title("RAM Distribution")
plt.show()

# Storage Distribution
sns.histplot(df['Storage'], kde=True)
plt.title("Storage Distribution")
plt.show()

# Final Price Distribution
sns.histplot(df['Final Price'], kde=True)
plt.title("Final Price Distribution")
plt.show()

############################################################################################################
#Regional/State/City Growth Analysis
#Now, analyze the growth trends in different regions, states, and cities.

#a) Growth Analysis:# Group by year and region to see the number of laptops sold
region_growth = df.groupby(['Year', 'Region']).size().unstack().fillna(0)
region_growth.plot(kind='line', marker='o')
plt.title("Growth in Laptop Sales by Region (2015-2023)")
plt.ylabel("Number of Laptops Sold")
plt.show()

# Group by year and state
state_growth = df.groupby(['Year', 'State']).size().unstack().fillna(0)
state_growth.plot(kind='line', marker='o', figsize=(12, 6))
plt.title("Growth in Laptop Sales by State (2015-2023)")
plt.ylabel("Number of Laptops Sold")
plt.show()

# Group by year and city
city_growth = df.groupby(['Year', 'City']).size().unstack().fillna(0)
city_growth.plot(kind='line', marker='o', figsize=(12, 6))
plt.title("Growth in Laptop Sales by City (2015-2023)")
plt.ylabel("Number of Laptops Sold")
plt.show()

############################################################################################################
#b) Penetration Analysis:
#Penetration can be understood by the proportion of sales in each region, state, and city 
#relative to the total sales.
# Penetration by Region
region_penetration = df['Region'].value_counts(normalize=True) * 100
sns.barplot(x=region_penetration.index, y=region_penetration.values)
plt.title("Market Penetration by Region")
plt.ylabel("Percentage of Sales")
plt.show()

# Penetration by State
state_penetration = df['State'].value_counts(normalize=True).head(10) * 100
sns.barplot(x=state_penetration.index, y=state_penetration.values)
plt.title("Top 10 Market Penetration by State")
plt.ylabel("Percentage of Sales")
plt.xticks(rotation=45)
plt.show()

# Penetration by City
city_penetration = df['City'].value_counts(normalize=True).head(10) * 100
sns.barplot(x=city_penetration.index, y=city_penetration.values)
plt.title("Top 10 Market Penetration by City")
plt.ylabel("Percentage of Sales")
plt.xticks(rotation=45)
plt.show()

############################################################################################################
#Define the Price Bands
#First, categorize the "Final Price" column into the specified price bands.
# Define the price bands
price_bins = [0, 25000, 50000, 75000, 100000, np.inf]
price_labels = ['₹0-₹25K', '₹25K-₹50K', '₹50K-₹75K', '₹75K-₹100K', '₹100K and above']

# Categorize 'Final Price' into these bins
df['Price Band'] = pd.cut(df['Final Price'], bins=price_bins, labels=price_labels)

# Display the first few rows to check the new 'Price Band' column
print(df.head())

#2. Visualize the Distribution of Laptops in Each Price Band
#You can visualize the distribution of laptops across these price bands.
# Distribution of Laptops in Each Price Band
sns.countplot(x=df['Price Band'])
plt.title("Distribution of Laptops in Each Price Band")
plt.ylabel("Number of Laptops")
plt.xlabel("Price Band")
plt.xticks(rotation=45)
plt.show()

#3. Analyze the Trends Over the Years
#Let's see how the distribution of sales in each price band has changed over the years.
# Group by Year and Price Band to analyze the trends
price_band_trends = df.groupby(['Year', 'Price Band']).size().unstack().fillna(0)

# Plotting the trend for each price band over the years
price_band_trends.plot(kind='line', marker='o', figsize=(12, 6))
plt.title("Price Band Trends Over the Years (2015-2023)")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Year")
plt.legend(title='Price Band')
plt.show()

#4. Analyze the Distribution Across Regions, States, and Cities
#let's see which regions, states, or cities prefer certain price bands.

#By Region:
# Price Band distribution by Region
region_price_band = df.groupby(['Region', 'Price Band']).size().unstack().fillna(0)

# Plotting
region_price_band.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Price Band Distribution by Region")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Region")
plt.legend(title='Price Band')
plt.show()

#By State:
# Price Band distribution by State
state_price_band = df.groupby(['State', 'Price Band']).size().unstack().fillna(0).head(10)  # Top 10 states

# Plotting
state_price_band.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Price Band Distribution by State (Top 10)")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("State")
plt.legend(title='Price Band')
plt.show()

#By City:
# Price Band distribution by City
city_price_band = df.groupby(['City', 'Price Band']).size().unstack().fillna(0).head(10)  # Top 10 cities

# Plotting
city_price_band.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Price Band Distribution by City (Top 10)")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("City")
plt.legend(title='Price Band')
plt.show()

# Interpreting the Results
# Overall Distribution: The bar plot will show you how laptops are distributed across different price bands.
# Trends Over Time: The line plots will help identify any shifts in consumer preference for different price bands over the years.
# Regional, State, and City Preferences: The stacked bar plots will show how the price band preferences differ by region, state, or city.

##############################################################################################################
#Gender Segment Analysis
#Analyze how different genders purchase laptops. You can explore the distribution of purchases across different price bands and regions.
# Count of purchases by gender
gender_distribution = df['Gender'].value_counts()

# Plotting the gender distribution
sns.barplot(x=gender_distribution.index, y=gender_distribution.values)
plt.title("Laptop Purchases by Gender")
plt.ylabel("Number of Purchases")
plt.xlabel("Gender")
plt.show()

# Price Band distribution by Gender
gender_price_band = df.groupby(['Gender', 'Price Band']).size().unstack().fillna(0)

# Plotting
gender_price_band.plot(kind='bar', stacked=True)
plt.title("Price Band Distribution by Gender")
plt.ylabel("Number of Purchases")
plt.xlabel("Gender")
plt.legend(title='Price Band')
plt.show()

# Region-wise distribution by Gender
gender_region = df.groupby(['Region', 'Gender']).size().unstack().fillna(0)

# Plotting
gender_region.plot(kind='bar', stacked=True)
plt.title("Region-wise Laptop Purchases by Gender")
plt.ylabel("Number of Purchases")
plt.xlabel("Region")
plt.legend(title='Gender')
plt.show()

#Top 5 Cities Segment Analysis
#Identify the top 5 cities based on sales and analyze the purchasing behavior within these cities.
# Identify top 5 cities based on total sales
top_5_cities = df['City'].value_counts().head(5).index

# Filter data for top 5 cities
top_cities_data = df[df['City'].isin(top_5_cities)]

# Distribution of sales across top 5 cities
sns.countplot(x=top_cities_data['City'])
plt.title("Laptop Purchases in Top 5 Cities")
plt.ylabel("Number of Purchases")
plt.xlabel("City")
plt.xticks(rotation=45)
plt.show()

# Price Band distribution in Top 5 Cities
top_city_price_band = top_cities_data.groupby(['City', 'Price Band']).size().unstack().fillna(0)

# Plotting
top_city_price_band.plot(kind='bar', stacked=True)
plt.title("Price Band Distribution in Top 5 Cities")
plt.ylabel("Number of Purchases")
plt.xlabel("City")
plt.legend(title='Price Band')
plt.xticks(rotation=45)
plt.show()


# Tier 2 & Tier 3 Cities Segment Analysis
# Assuming you have a way to classify cities into Tier 2 and Tier 3, analyze these segments separately. If you don't have this classification in the data, you might need to manually assign tiers based on city size/population.

# a) Create a Tier Classification
# If city tier classification is not in your dataset, you can create a simple one for demonstration:

    # Example classification - You'll need a real classification for accuracy
tier_2_cities = ['Jaipur', 'Lucknow', 'Pune']  # Example cities
tier_3_cities = ['Bhubaneswar', 'Patna', 'Kanpur']  # Example cities

df['City Tier'] = np.where(df['City'].isin(tier_2_cities), 'Tier 2', 
                           np.where(df['City'].isin(tier_3_cities), 'Tier 3', 'Others'))

#b) Analyze Tier 2 & Tier 3 Cities
# Filter Tier 2 and Tier 3 cities
tier_2_3_data = df[df['City Tier'].isin(['Tier 2', 'Tier 3'])]

# Distribution of sales in Tier 2 & Tier 3 cities
sns.countplot(x=tier_2_3_data['City Tier'])
plt.title("Laptop Purchases in Tier 2 & Tier 3 Cities")
plt.ylabel("Number of Purchases")
plt.xlabel("City Tier")
plt.show()

# Price Band distribution in Tier 2 & Tier 3 cities
tier_price_band = tier_2_3_data.groupby(['City Tier', 'Price Band']).size().unstack().fillna(0)

# Plotting
tier_price_band.plot(kind='bar', stacked=True)
plt.title("Price Band Distribution in Tier 2 & Tier 3 Cities")
plt.ylabel("Number of Purchases")
plt.xlabel("City Tier")
plt.legend(title='Price Band')
plt.show()

# Interpreting the Results
# Gender Analysis: Understand how male and female customers differ in their purchasing behavior, particularly in terms of price preferences and regional differences.
# Top 5 Cities: Gain insights into the dominant price bands and potential city-specific trends in your top markets.
# Tier 2 & Tier 3 Cities: Analyze how customer behavior differs in smaller cities, which can help tailor marketing strategies to these segments.
# By analyzing these customer segments, you can better understand the preferences and behaviors of different groups within your customer base. This knowledge can help in tailoring your marketing, sales, and product strategies to meet the needs of these segments more effectively.


#############################################################################################################################
# Sales Channel Distribution
#First, let's examine how sales are distributed across the different channels.
# Count of sales by channel
channel_distribution = df['Retailer Type'].value_counts()

# Plotting the distribution of sales channels
sns.barplot(x=channel_distribution.index, y=channel_distribution.values)
plt.title("Distribution of Laptop Sales by Channel")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Sales Channel")
plt.xticks(rotation=45)
plt.show()

#Price Band Distribution by Sales Channel
#Next, let's analyze how different price bands perform across different sales channels.
# Price Band distribution by Sales Channel
channel_price_band = df.groupby(['Retailer Type', 'Price Band']).size().unstack().fillna(0)

# Plotting
channel_price_band.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Price Band Distribution by Sales Channel")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Sales Channel")
plt.legend(title='Price Band')
plt.xticks(rotation=45)
plt.show()

#Sales Trends Over Time
#Let's analyze how the contribution of each sales channel has changed over time.
# Group by Year and Sales Channel to analyze trends
channel_trends = df.groupby(['Year', 'Retailer Type']).size().unstack().fillna(0)

# Plotting the trends over time
channel_trends.plot(kind='line', marker='o', figsize=(12, 6))
plt.title("Sales Trends by Channel Over the Years (2015-2023)")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Year")
plt.legend(title='Sales Channel')
plt.show()

#Regional Preferences by Sales Channel
#now analyze how regional preferences vary by sales channel.
# Region-wise distribution by Sales Channel
region_channel = df.groupby(['Region', 'Retailer Type']).size().unstack().fillna(0)

# Plotting
region_channel.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Region-wise Laptop Sales by Channel")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("Region")
plt.legend(title='Sales Channel')
plt.xticks(rotation=45)
plt.show()

#Top Cities by Sales Channel
#Analyze how the top cities perform across different sales channels.
# Filter data for top 5 cities
top_5_cities = df['City'].value_counts().head(5).index
top_cities_data = df[df['City'].isin(top_5_cities)]

# Group by City and Sales Channel
city_channel = top_cities_data.groupby(['City', 'Retailer Type']).size().unstack().fillna(0)

# Plotting
city_channel.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Top 5 Cities: Sales Distribution by Channel")
plt.ylabel("Number of Laptops Sold")
plt.xlabel("City")
plt.legend(title='Sales Channel')
plt.xticks(rotation=45)
plt.show()

# Interpreting the Results
# Overall Distribution: The bar plot of sales channels will show which channels are most popular among customers.
# Price Band Analysis: The stacked bar plot will reveal which price bands are more popular on each channel, helping you understand customer behavior across different sales channels.
# Trends Over Time: The line plot will help you track how the popularity of each sales channel has changed over the years, indicating shifts in customer buying preferences.
# Regional Preferences: The region-wise analysis can provide insights into whether certain regions prefer online shopping or physical stores.
# City Preferences: Analyzing top cities will help you identify if there is a city-specific preference for certain sales channels.
# This Sales Channel Analysis will give you a comprehensive understanding of how customers are engaging with different sales channels and how you can optimize your sales strategy to better cater to their preferences.

#######################################################################################################################################


#Trend Analysis
#Before diving into predictive modeling, perform a trend analysis to understand the historical growth patterns.
# Aggregate sales data by year
yearly_sales = df.groupby('Year')['Final Price'].sum()

# Plotting the historical sales trend
plt.figure(figsize=(10, 6))
plt.plot(yearly_sales.index, yearly_sales.values, marker='o', linestyle='-', color='b')
plt.title("Historical Sales Trend (2015-2023)")
plt.xlabel("Year")
plt.ylabel("Total Sales (in INR)")
plt.grid(True)
plt.show()
###########################################################################################################################
#DATA PREPROCESSING 


# Display the first few rows of the dataset
print(df.head())

# Display the summary statistics of the dataset
print(df.describe())

# Display information about the dataset (e.g., column types, missing values)
print(df.info())

# Check for missing values in the dataset
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Drop rows where essential columns like 'Final Price' or 'Year' are missing
df = df.dropna(subset=['Final Price', 'Year'])

# Fill or drop any other columns with missing values if necessary
# For simplicity, we could fill missing values in categorical columns with the mode
df['Region'].fillna(df['Region'].mode()[0], inplace=True)
df['State'].fillna(df['State'].mode()[0], inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)
df['Retailer Type'].fillna(df['Retailer Type'].mode()[0], inplace=True)

# If GPU is not critical for analysis, we could drop it, otherwise, fill missing with a placeholder
df['GPU'].fillna('Unknown', inplace=True)

# Remove any potential outliers in the 'Final Price' column if necessary
# Here, we'll remove rows where the 'Final Price' is an extreme outlier
df = df[df['Final Price'] > 0]  # Assuming that Final Price must be positive

# After cleaning, let's check the final shape of the dataframe
print("Final DataFrame shape:", df.shape)
print(df.head())

from sklearn.preprocessing import StandardScaler
# Standardize numerical features
scaler = StandardScaler()

# Selecting numerical columns to standardize (e.g., 'Final Price', 'RAM', 'Storage')
numerical_features = ['Final Price', 'RAM', 'Storage']  # Add more features if necessary

# Standardizing the features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Display the first few rows of the standardized dataframe
print(df.head())
##########################################################################################################
# Now that the data is clean, let's proceed to analysis and prediction.







