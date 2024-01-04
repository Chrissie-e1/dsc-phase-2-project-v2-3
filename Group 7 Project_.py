#!/usr/bin/env python
# coding: utf-8

# # Northwestern County: Real Estate Market Analysis  

# ![PHOTO-2024-01-04-12-32-15.jpg](attachment:PHOTO-2024-01-04-12-32-15.jpg)

# ### Project Overview:   
# Every Door Real Estate, our stakeholder, is facing a business
# issue related to their clients' home-buying process. The objective is to help clients comprehend more about the house attributes and make informed decisions in their home-buying/selling process. The challenge is to determine a model that can effectively
# demonstrate how property attributes impact house prices. This model will help Every
# Door's agents to provide their clients with a clearer insight into how much pre-approval they need for their ideal property.

# ### Audience:
# Every Door Real Estate. The agency's mission is to provide clients with comprehensive insights to enhance their understanding on the correlation between house attributes and house prices.
# 

# ### Goal:  
# This analysis will be guiding Every Door Real Estate on the impact of various property attributes on
# their pricing structure. The anlysis aims to offer a clear understanding of how property attributes collectively influence the estimated value of homes, enabling strategic decisions for optimal return on investment.

# ## Data Understanding:Dataset
# This analysis uses King County House Sales dataset, found in kc_house_data.csv in the data folder in GitHub repository. The dataset is significant in predicting house prices in relation to property attributes.  
# The data is imported and uploaded using necessary libraries. Some of the libraries imported for data analysis includes matplotlib, numpy,pandas, seaborn, scipy,and others. The structure of the data is made up of 21597 observations and 21 columns describing the property attributes.

# ## Column Names and Descriptions for King County Data Set
# 1. **Id** - Unique identifier for a house  
# 2. **Date** - Date house was sold  
# 3.**Price** - Sale price (prediction target)  
# 4.**Bedrooms** - Number of bedrooms  
# 5.**Bathrooms** - Number of bathrooms  
# 6.**Sqft_living** - Square footage of living space in the home  
# 7. **Sqft_lot** - Square footage of the lot  
# 8. **Floors** - Number of floors (levels) in house  
# 9. **Waterfront** - Whether the house is on a waterfront  
# Includes Duwamish, Elliott Bay, Puget Sound, Lake Union, Ship Canal, Lake Washington, Lake Sammamish, other lake, and river/slough waterfronts
# 10. **View** - Quality of view from house  
# Includes views of Mt. Rainier, Olympics, Cascades, Territorial, Seattle Skyline, Puget Sound, Lake Washington, Lake Sammamish, small lake / river / creek, and other
# 11. **Condition** - How good the overall condition of the house is. Related to maintenance of house.  
# See the King County Assessor Website for further explanation of each condition code
# 12. **Grade** - Overall grade of the house. Related to the construction and design of the house.  
# See the King County Assessor Website for further explanation of each building grade code
# 13. **Sqft_above** - Square footage of house apart from basement  
# 14. **Sqft_basement** - Square footage of the basement  
# 15. **Yr_built** - Year when house was built  
# 16. **Yr_renovated** - Year when house was renovated  
# 17. **Zipcode** - ZIP Code used by the United States Postal Service  
# 18. **Lat** - Latitude coordinate  
# 19. **Long** - Longitude coordinate  
# 20. **Sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors  
# 21. **Sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors

# The data suggest that prices of houses depends on the features of the property.
# However data cleaning and wrangling is important to make the data more useful. The process of data cleaning and wrangling includes checking for missing values, duplicates, uniformity,and data formats.Additional cleaning including replacing missing values, dropping insignificant data, filling null values, and dropping duplicates using relevant functions and methods is also done to further make the data more valuable.
# 

# # Data Loading and Importation of Modules

# In[ ]:


#Import necessary libraries
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("kc_house_data.csv")
df


# # Data Cleaning and Wrangling:Data Preparation

# The process of data cleaning and wrangling entailed formatting the data to remove duplicates, filled missing values, deleting undesirable symbols, and selecting required property variables for further modelling. From the observations, the data did not have any duplicates.  
# However, the variables:  
# a.	Waterfront, yr_renovated, and view contained missing values which were filled with appropriate data. The missing values in waterfront were replaced with “NO” to assume the households lacked waterfronts. The missing values in view were replaced with “NONE” to assumed a lack of view. The missing values in yr_renovated were replaced with 0 to mean the houses were never renovated.   
# b.	The values in the variable bathroom were standardized to decimal places because it was observed the values lacked uniformity of decimal places.   
# c.	The date section was standardized to: day, month, and year format for uniformity.    
# d.	The sqft_basement contained undesirable symbols which were removed, replaced with mean, and the data converted to numerical values. The data was also converted to 2 decimal places for uniformity.    
# e.	The values in the latitude and longitude variables were converted to 2 decimal places for effective analysis and modelling.    
# f.	The whole data set was confirmed clean and ready for modelling using the info() method.    
# g.	During wrangling, the columns 'zipcode','id', 'lat', 'long', 'yr_renovated', 'yr_built' were observed to be insignificant and consequently were dropped.     
# 

# In[ ]:


#observe data structure
df.shape


# In[ ]:


#check for duplicates
df.duplicated().sum()


# In[ ]:


#confirm duplicates
df[df.duplicated(keep=False)]


# In[ ]:


#Check for nulls
df.isna().sum()


# In[ ]:


#check data type and unique values in waterfront;
df["waterfront"].unique()


# In[ ]:


#check data type and unique values in waterfront
df["waterfront"].value_counts()


# In[ ]:


#replace nan with the mode, that is No value
#Assumption is nan means no waterfront
#Check for successful replacements
df['waterfront'].replace(np.nan, "NO", inplace=True, regex=False)


# In[ ]:


#check for successful replacement
df["waterfront"].unique()


# In[ ]:


#check data type and unique values in conditions
df["condition"].unique().tolist()


# In[ ]:


#check data type and unique values in conditions
df["condition"].value_counts()


# In[ ]:


#check data type and unique values in bathrooms
#output shows inconsistent decimals
df["bathrooms"].unique()


# In[ ]:


#Round to 2 decimals for uniformity
df["bathrooms"]=df["bathrooms"].round(2)
df["bathrooms"]


# In[ ]:


#Let's assume that missing values in 'yr_renovated'
#mean no renovations were done, so we'll fill the missing values with 0.
# Fill missing values in 'yr_renovated' with 0
df['yr_renovated'].fillna(0, inplace=True)

# Verify the changes
print(df['yr_renovated'].value_counts())


# In[ ]:


#check data type and unique values in view
df["view"].unique().tolist()


# In[ ]:


#check data type and unique values in conditions
view_counts = df['view'].value_counts()
view_counts


# In[ ]:


# Replacing the NaN values with mode NONE
# Verifying the Changes
df['view'].replace(np.nan, "NONE", inplace=True, regex=False)
df['view'].unique()


# In[ ]:


# changing my date column for easier readability
df['date'] = pd.to_datetime(df['date'],format='%m/%d/%Y')


# In[ ]:


#check for unique values in sqft_basement
df['sqft_basement'].unique()


# In[ ]:


# replace '?' with NaN in sqft_basement
df['sqft_basement'].replace('?', np.nan, inplace=True)

# convert the column to a numeric data type
df['sqft_basement'] = pd.to_numeric(df['sqft_basement'])

# fill missing values with the mean value of the column
df['sqft_basement'].fillna(np.mean(df['sqft_basement']), inplace=True)


# In[ ]:


#check data type and unique values in 'sqft_basement'
df['sqft_basement'].unique()


# In[ ]:


#replace symbols in sqft_basement
df['sqft_basement']=df['sqft_basement'].replace({',': ''}, regex=True)
df['sqft_basement']


# In[ ]:


#round values to 2 decimals places for uniformity
df["sqft_basement"]=df["sqft_basement"].round(2)
df["sqft_basement"]


# In[ ]:


#check for unique values in latitude
df['lat'].unique()


# In[ ]:


#round values to 2 decimals places for uniformity
df["lat"]=df["lat"].round(2)
df["lat"]


# In[ ]:


#check for uniformity in longitude
df['long'].unique()


# In[ ]:


#round values to 2 decimals places for uniformity
df["long"]=df["long"].round(2)
df["long"]


# In[ ]:


#check for unique values in zipcode
df['zipcode'].unique()


# In[ ]:


#drop unneccessary columns, less significant
df.drop(['id','zipcode', 'lat', 'long', 'yr_renovated', 'yr_built'],
        axis=1, inplace=True)


# In[ ]:


#confirm columns dropped
df.head(4)


# In[ ]:


#understand structure of dataset
df.info()


# # Data Exploration
# 

# In the further exploring the data, the required columns were converted from object to integer data type.This was the case with sqft_basement. A column, total_space, was created to include sqft_living and sqft_lot variables. An additional season column was added to analyse how housing prices across the different seasons. A further analysis was conducted using the describe()method.
# The data was then check for correlation visually and statistically to determined relevant columns for use in modelling. When checking for correlation, variables than were greater than .5 were considered more correlated to price (with a correlation of 1.0). These variables were retained for machine learning and modelling through a multiple regression analysis. Additionally, the selected variables were checked for skewness. All the variables were skewed and the data was standardized to normal distribution. The outliers were checked and removed to make the regression model for effective. Lastly, variables with categorical data were converted to dummy variables and the data added to the dataframe.

# In[ ]:


#check for columns
df.columns


# In[ ]:


#check data types
df.dtypes


# In[ ]:


#convert to numeric from object data type
df['sqft_basement']=pd.to_numeric(df['sqft_basement'].replace('?', '0')).astype(int)
df['sqft_basement']


# **Feature Engineering**

# In[ ]:


#merge 'sqft_living' and 'sqft_lot' columns
df['total_space'] = df['sqft_living'] + df['sqft_lot']
df['total_space']


# In[ ]:


#Create column season to check house prices in different seasons
def get_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'


# In[ ]:


# Create a new 'season' column by applying the function to the 'date' column
df['season'] = df['date'].dt.month.apply(get_season)


# In[ ]:


#check dataframe
df


# # Data Analysis

# In[ ]:


# Check data summary
df.describe()


# In[ ]:


#check for correlations
df.corr()["price"].sort_values(ascending=False)


# In[ ]:


#Visualize correlations
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True, cmap="YlGnBu");


# In[ ]:


#drop less correlated variables
df.drop(['bedrooms','sqft_lot', 'floors', 'sqft_basement', 'sqft_lot15', 'total_space'],
        axis=1, inplace=True)


# In[ ]:


#confirm changes in the dataframe
df.head(2)


# In[ ]:


from sklearn.model_selection import train_test_split
#check for basic correlations in the data
df.hist(figsize=(30,20), color='grey', edgecolor='white');


# ### Deep Processing
# Checking and Removing Skewness

# In[ ]:


#Visualize skewness using a histogram
df.hist(figsize=(30,20));


# In[ ]:


#standardize data to normal distribution
df['price']=np.log(df['price']+1)
df['bathrooms']=np.log(df['bathrooms']+1)
df['sqft_living']=np.log(df['sqft_living']+1)
df['sqft_above']=np.log(df['sqft_above']+1)
df['sqft_living15']=np.log(df['sqft_living15']+1)


# In[ ]:


#confirm successful data standardization
df.hist(figsize=(30,20));


# In[ ]:


#Check for distribution using displot
sns.displot(df,bins=20,kde=True);


# In[ ]:


#Bathroom variable shows abnormal distribution
#Remove for outliers in the bathroom column
import numpy as np
count = 0
bathroom_outliers = []
mean = np.mean(df['bathrooms'])
max_distance = np.std(df['bathrooms']) * 3

for idx, row in df['bathrooms'].T.items():
    if abs(row-mean) >= max_distance:
        count += 1
        df.drop(idx, inplace=True)
count


# In[ ]:


# Assuming df is your DataFrame with cleaned data
#Confirm outliers in boxplot
min_values, max_values = df.bathrooms.quantile([0.010, 0.95])
min_values, max_values
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['bathrooms'])
plt.title('Box Plot of Bathrooms')
plt.show()


# In[ ]:


# creating summary statistics of pricing of each season
# Group by 'season' and calculate summary statistics
df.groupby('season')['price'].agg(["count","mean"])


# From the summary, we have more average sales during spring evidently seen in value counts and mean.

# In[ ]:


#Visualization: relationship between price and seasons
# Plotting seasonal trends using a bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='season', y='price', data=df)
plt.title('Seasonal Trends in Housing Prices')
plt.xlabel('Season')
plt.ylabel('Average Price')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 6))
sns.barplot(x='waterfront', y='price', data=df)
plt.title('Bar Graph of Price vs. Waterfront')
plt.xlabel('Waterfront (1: Yes, 0: No)')
plt.ylabel('Average Price')
plt.show()


# We see that properties on a waterfront typically have higher sale prices than properties that are inland

# In[ ]:


plt.figure(figsize=(8, 6))
sns.barplot(x='view', y='price', data=df)
plt.title('Bar Graph of Price vs. View')
plt.xlabel('View ( 0:NONE, 1:GOOD, 2:EXCELLENT, 3:AVERAGE, 4:FAIR)')
plt.ylabel('Average Price')
plt.show()


# We see that properties with an excellent view typically have higher sale prices

# In[ ]:


plt.figure(figsize=(18, 6))
sns.barplot(x='grade', y='price', data=df)
plt.title('Bar Graph of Price vs. grade')
plt.xlabel('View (7: Average, 6: Low Average, 8: Good, 11: Excellent, 9: Better,5: Fair, 10: Very Good, 12: Luxury, 4: Low, 3: Poor,13: Mansion)')
plt.ylabel('Average Price')
plt.show()


# We see that properties graded mansion, luxury,excellent, and very good typically have higher sale prices

# In[ ]:


plt.figure(figsize=(12, 6))
sns.barplot(x='condition', y='price', data=df)
plt.title('Bar Graph of Price vs. condition')
plt.xlabel('Condition (1: Average, 2: Very Good, 3: Good, 4:Poor, 5:Fair)')
plt.ylabel('Average Price')
plt.show()


# We see that properties with very good condition typically have higher sale prices than properties that are average, good, poor, and fair condition

# ### Analysing and Combined Modelling

# #  Regression Models  
# Here we begin to examine potential relationships between different combinations of variables. We are looking to see if we can build a model that shows strong relationships between our variables of interest.

# Now that the categorical variables are removed, let us check for correlation of independent vs dependent variables.    
# In this case, our dependent variable being "price"  
# 

# In[ ]:


#Correlations
df.corr()["price"]


# Correlation is a measure of causal relationship rather than causation  
# In this summary above, the variables show low-to-medium-to-strong correlations with price  
# The property value with a highest correlation with price is the sqft_living followed by sqft_living15.
# 

# **Modelling for a simple Linear Regression**  
# Regression models are evaluated against a "baseline".  
# For simple linear regression, this baseline is an "intercept-only" model that just predicts the mean of the dependent variable every time.  
# For multiple linear regression, build a simple linear regression to be that baseline.    
# 
# 
# In this case, the simple linear regression will be based price vs sqft_living.  
# sqft_living showed the highest and strongest correlation hence fit to create a baseline model for the multiple linear regression
# 
# In other words, set up the formula
# 
# 𝑦̂ =𝛽0^+𝛽1^𝑥
# 
# Where  𝑦̂
#   is price, the dependent (endogenous) variable, and  𝑥
#   is sqft_living, the independent (exogenous) variable. When we fit our model, we are looking for  𝛽1^
#   (the slope) and  𝛽0^
#   (the intercept).

# **Baseline_model**

# In[ ]:


#model 1
#Price vs Sqft_living
X="sqft_living"
y="price"
#plot a scatter plot to show sqft_living vs price
df.plot(x="sqft_living", y="price", kind="scatter", title="Price Vs Sqft_living");


# In[ ]:


#fit the baseline_model since the scatter shows correlation
X=df["sqft_living"]
y=df["price"]
baseline_model = sm.OLS(endog=y, exog=sm.add_constant(X))
baseline_model


# In[ ]:


baseline_modelresults = baseline_model.fit()
baseline_modelresults


# In[ ]:


#Inferential analysis using summary for baseline_modelresults
baseline_modelresults.summary()


# 
# ### Simple Linear Regression Results and Interpretation
# 
# Looking at the summary above,   
# 
# a. Constant/y-intercept/C=6.7597  
# b. slope/m/x-intercept=0.8327  
# c. therefore, regression line is given by:  
# 
# $$ \hat{price} = 6.7597 + 0.8327  sqft\_living $$
# 
# * The model is statistically significant overall, with an F-statistic p-value well below 0.05
# * The model explains (R-squared value= 0.493) 50% of the variance in Price explained by sqft_living
# * The model coefficients (`const` and `sqft_Living`) are both statistically significant, with t-statistic p-values well below 0.05
# * If a house had sqft_living of 0 sqft, we would expect the price to be about 6.7597K (thousands of dollars)
# * For each additional 1sqft living of a property, we see an associated increase in price of about 0.8327K (thousands of dollars)
# 
# Note that all of these coefficients represent associations rather than causation.

# # Simple Linear Regression Visualization¶
# We'll also plot the actual vs. predicted values:

# In[ ]:


#Show the model
#For Regression line
fig, ax = plt.subplots()
df.plot.scatter(x="sqft_living", y="price", label="Data points", ax=ax)
sm.graphics.abline_plot(model_results=baseline_modelresults,
                        label="Regression line",
                        color="black",
                        ax=ax)
ax.legend();


# In[ ]:


sm.graphics.plot_fit(baseline_modelresults, "sqft_living")
plt.show()


# In[ ]:


#Calculate residues
baseline_modelresults.resid


# In[ ]:


#Visualize residues
fig, ax = plt.subplots()

ax.scatter(df["sqft_living"], baseline_modelresults.resid)
ax.axhline(y=0, color="black")
ax.set_xlabel("sqft_living")
ax.set_ylabel("residuals");



# In[ ]:


#show residue using histogram
fig, ax = plt.subplots(figsize=(15,5))
ax.hist(baseline_modelresults.resid)
ax.set_ylabel=("Price")
ax.set_xlabel=("sqft_living")
ax.set_title("Distribution of Residuals (sqft_living)");


# In[ ]:


#Residue plot using qq-plot
from scipy.stats import norm
fig, ax = plt.subplots(figsize=(10,8))
sm.graphics.qqplot(baseline_modelresults.resid, dist=norm, line="45", fit=True, ax=ax)
ax.set_title("Quantiles of Residuals (sqft_living)")
plt.show()


# **Model evaluation**

# In[ ]:


mae = baseline_modelresults.resid.abs().sum() / len(y)
mae


# 
# model is off by 0.31K in any given prediction

# In[ ]:





# # SECOND MODEL:..with additional independent variables

# The second model employs variables with strong causal relationship with price in reference to the correlation summary. The variables added in the second model include bathrooms, sqft_above, and sqft_living15.

# In[ ]:


X_secondvariable=df[["sqft_living","bathrooms","sqft_above","sqft_living15"]]
X_secondvariable.head(3)


# In[ ]:


#for second model and Summary
second_model = sm.OLS(y, sm.add_constant(X_secondvariable))
second_modelresults = second_model.fit()

print(second_modelresults.summary())


# # Second Model:Results and Interpretations
# 
# 
# Model Regression equation:
# 
# $$ \hat{Price} = 5.7587 + 0.6355 sqft\_living+ 0.0799 bathrooms - 0.0764 sqft\_above + 0.3935 sqft\_living15 $$
# 
# The coeffecient has decreased from 6.7597 in the first model to 5.7587 in the second model. This is because of the intercept is now with respect to bathrooms, sqft_living15, and sqft_above  
# * The model is statistically significant overall, with an F-statistic p-value well below 0.05
# * The model explains about 43% of the variance in price
# * The model coefficients (`const`, `bathrooms`, `sqft_above`and `sqft_living15`) are all statistically significant, with t-statistic p-values well below 0.05
# * If a house had sqft_above of 0 sqft, or sqft_living15 of 0, or no bathroom, we would expect the price to be about 5.7587K (thousands of dollars)
# * For each additional 1sqft_living15, we see an associated increase in price of about 0.3935K
# * For each additional 1sqft_above, we see an associated decrease in price of about 0.0764K
# * For each additional 1bathroom, we see an associated increase in price of about 0.0799 (thousands of dollars)
# * For each additional 1sqft_living, we see an associated increase in price of about 0.6355 (thousands of dollars)  
#     **This is a little bit smaller of a decrease than we saw with the simple model, but not a big change. This means that second model was not meaningfully confounding in the relationship between price and sqft_living**  
# Note that all of these coefficients represent associations rather than causation.
# * For each increase of 1 lb in car weight, we see an associated decrease in MPG of about .007
#   * This is a little bit smaller of a decrease than we saw with the simple model, but not a big change. This means that model year was not meaningfully confounding in the relationship between weight and MPG
# 

# # Visualization for the second model

# a. **Model fit**

# In[ ]:


sm.graphics.plot_fit(second_modelresults, "sqft_living")
plt.show()


# This shows the true (blue) vs. predicted (red) values, with the particular predictor (in this case, sqft_living) along the x-axis. Note that unlike with a simple regression, the red dots don't appear in a perfectly straight line. This is because their predictions are made based on the entire model, not just this predictor.
# 

# b.**Fit Model for Bathrooms, Sqft_above, and Sqft_living15**

# In[ ]:


#Bathroom
sm.graphics.plot_fit(second_modelresults, "bathrooms")
plt.show()


# In[ ]:


#For Sqft_above
sm.graphics.plot_fit(second_modelresults, "sqft_above")
plt.show()


# In[ ]:


#For sqft_living15
sm.graphics.plot_fit(second_modelresults, "sqft_living15")
plt.show()


# C. **Partial Regression Plot**  
# Then, instead of a basic scatter plot with a best-fit line (since our model is now higher-dimensional), we'll use two partial regression plots, one for each of our predictors.

# In[ ]:


fig = plt.figure(figsize=(15,10))
sm.graphics.plot_partregress_grid(second_modelresults,
                                  exog_idx=["bathrooms", "sqft_living15","sqft_living", "sqft_above"],
                                  fig=fig)
plt.tight_layout()
plt.show()


# Plot shows a linear relationship with a non-zero slope, it is therefore beneficial to add sqft_living to the model, vs. having a model without sqft_living (i.e. a model with just an intercept and the other variables).
# 
# It can be deduced that these predictors are useful and should be included in the model. However, bathroom and sqft_living can be ignored since their slopes are nearly zeros slopes

# d.**Residual Plots combination other plots**

# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(second_modelresults, "sqft_living", fig=fig)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(second_modelresults, "sqft_above", fig=fig)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(second_modelresults, "bathrooms", fig=fig)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(second_modelresults, "sqft_living15", fig=fig)
plt.show()


# **Model Evalution**

# In[ ]:


mae = second_modelresults.resid.abs().sum() / len(y)
mae


# Model is off by 0.306k price for a given prediction

# # Third Model: Multiple Regression with Many Features

# In[ ]:


# dropping price because this is our target,
#then only selecting numeric features
X_all = df.drop(["price"], axis=1).select_dtypes("number")
X_all.head(5)


# In[ ]:


third_model = sm.OLS(y, sm.add_constant(X_all))
third_results = third_model.fit()

print(third_results.summary())


# # Third model results and Interpretations

# The model is:  
# 
# price = 9.9275 − 0.0510bedrooms - 0.0022bathrooms + 0.0771sqft\_living - 0.0003sqft\_lot + 0.1141floors - 0.1283sqft\_above + 0.00007104sqft\_basement + 0.3745sqft\_living15-0.000001045sqft\_lot15 + 0.003total\_space  
#  *The model is statistically significant overall, with an F-statistic p-value well below 0.05  
#  *The model explains about 51.8% of the variance in price  
#  *Using multiple predictors increased the value of R-Squared by 15.11%, this makes this model fit and suitable for use  
#  *Only some of the model coefficients are statistically significant while others statistically insignificant  
#  *Bedrooms, sqft\_lot, floors, sqft\_above, sqft\_basement, sqft\_living15, sqft\_lot15, and total\_space p-values below 0.05 and are therefore statistically significant  
#  *Bathrooms and sqft\_living have p-values above 0.05. This means that there is greater than a 1 in 20 chance that their true coefficients are 0 (i.e. they have no effect on price), and are thus not statistically significant at an alpha of 0.05  
# 

# **Visualization using Partial regression plots**

# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_partregress_grid(third_results,
                                  exog_idx=list(X_all.columns.values),
                                  grid=(2,4),
                                  fig=fig)
plt.show()


# **Model Evaluation**

# In[ ]:


mae = third_results.resid.abs().sum() / len(y)
mae


# model is off by 0.29K dollars in a given prediction

# model value is smaller and near zero making it the best model

# # Modeling with categorical variables

# **Model with multiple  variables and waterfront being categorical variable**

# In[ ]:


# Model with multiple independent  variables and waterfront being categorical
df['waterfront'].value_counts(ascending=True)


# In[ ]:


# choosing my refence category based on the model to avoid dummy variable trap
X_waterfront_model = df[['sqft_living','bathrooms','sqft_above','sqft_living15','waterfront']]
# origin is categorical and needs to be numeric to run regression
X_waterfront_model = pd.get_dummies(X_waterfront_model, columns=['waterfront'], drop_first=True, dtype=int)

waterfront_model = sm.OLS(y, sm.add_constant(X_waterfront_model))
waterfront_results = waterfront_model.fit()

print(waterfront_results.summary())


# **Interpretation**

# sqft_living: For each additional square foot of living space, the estimated house price increases by 0.6225 thousand dollars.
# 
# bathrooms: For each additional bathroom, the estimated house price increases by 0.0845 thousand dollars.
# 
# sqft_above: For each additional square foot above ground, the estimated house price decreases by 0.0700 thousand dollars.
# 
# sqft_living15: For each additional square foot of the 2015 living space, the estimated house price increases by 0.3846 thousand dollars.
# 
# waterfront_YES: If a house has a waterfront (compared to not having a waterfront), the estimated house price increases by 0.7234 thousand dollars.
# 
# Overall Model:
# 
# The overall model has an R-squared value of 0.489, indicating that approximately 48.9% of the variability in house prices is explained by the model.
# Statistical Significance:
# 
# All coefficients have p-values less than 0.05, indicating that they are statistically significant.

# **Visualization**

# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
df.groupby("waterfront").mean('price').sort_values(by="price").plot.bar(y="price", ax=ax)
ax.axhline(y=df["price"].mean(), label="mean", color="black", linestyle="--")
ax.legend();


# Visualization using Partial regression plot

# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_partregress_grid(waterfront_results,
                                  exog_idx=list(X_all.columns.values),
                                  grid=(2,4),
                                  fig=fig)
plt.show()


# **Model Evaluation using mean absolute error

# In[ ]:


mae = waterfront_results.resid.abs().sum() / len(y)
mae


# model is off by 0.30K dollars in a given prediction

# **Model with multiple independent variables and view being: categorical variable**

# In[ ]:


# Choosing my reference category based on the one which is most common
df['view'].value_counts(ascending=True)


# In[ ]:


X_view_model = df[['sqft_living', 'bathrooms', 'sqft_above', 'sqft_living15', 'view']]

X_view_model = pd.get_dummies(X_view_model, columns=['view'], dtype=int)
#Drop one of the dummy variable columns ('view_NONE')

X_view_model = X_view_model.drop("view_NONE", axis=1)

X_view_model = sm.add_constant(X_view_model)

y = df['price']

View_model = sm.OLS(y, X_view_model)

View_results = View_model.fit()

print(View_results.summary())




# **Interpretation**
# 
# sqft_living: For each additional square foot of living space, the estimated house price increases by 0.5606 thousand dollars.
# 
# bathrooms: For each additional bathroom, the estimated house price increases by 0.0949 thousand dollars.
# 
# sqft_above: For each additional square foot above ground, the estimated house price decreases by 0.0152 thousand dollars.
# 
# sqft_living15: For each additional square foot15, the estimated house price increases by 0.3286 thousand dollars.
# 
# Excellent_view: If a house has an Excellent view (compared to not having one, which is represented by the dropped category 'view_None'), the estimated house price increases by 0.5959 thousand dollars.
# 
# Average_view:  If a house has an Average view (compared to not having one), the estimated house price increases by 0.2063 thousand dollars.
# 
# Fair_view:  If a house has a Fair view (compared to not having one), the estimated house price increases by 0.2479 thousand dollars.
# 
# Good_view:  If a house has an Good view (compared to not having one), the estimated house price increases by 0.2797 thousand dollars.
# 
# Overall Model:
# 
# The overall model has an R-squared value of 0.506, indicating that approximately 50.6% of the variability in house prices is explained by the model. Statistical Significance:
# 
# All coefficients have p-values less than 0.05, indicating that they are statistically significant expect for sqft_above

# **Visualization**

# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
df.groupby("view").mean('price').sort_values(by="price").plot.bar(y="price", ax=ax)
ax.axhline(y=df["price"].mean(), label="mean", color="black", linestyle="--")
ax.legend();


# Visualization using Partial regression plot

# In[ ]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_partregress_grid(View_results,
                                  exog_idx=list(X_all.columns.values),
                                  grid=(2,4),
                                  fig=fig)
plt.show()


# Model Evaluation using mean absolute error

# In[ ]:


mae = View_results.resid.abs().sum() / len(y)
mae


# The model is off by 0.30 thousand dollars in a given prediction
