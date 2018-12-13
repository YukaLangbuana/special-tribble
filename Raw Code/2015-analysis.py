#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats


# ## Government consumption by % GDP

# In[2]:


GOVERNMENT_CONSUMPTION_DF = pd.read_csv("govt_consumption_by_gdp.csv")
GOVERNMENT_CONSUMPTION_DF = GOVERNMENT_CONSUMPTION_DF.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
GOVERNMENT_CONSUMPTION_DF = GOVERNMENT_CONSUMPTION_DF.T
GOVERNMENT_CONSUMPTION_DF.columns = GOVERNMENT_CONSUMPTION_DF.iloc[0]
GOVERNMENT_CONSUMPTION_DF = GOVERNMENT_CONSUMPTION_DF.drop(["Country Name"])
GOVERNMENT_CONSUMPTION_DF = GOVERNMENT_CONSUMPTION_DF.apply(pd.to_numeric, errors='coerce')
GOVERNMENT_CONSUMPTION_DF


# ## Government Spending by % GDP

# In[3]:


GOVERNMENT_SPENDING_DF = pd.read_csv("govt_spending_by_gdp.csv")
GOVERNMENT_SPENDING_DF = GOVERNMENT_SPENDING_DF.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
GOVERNMENT_SPENDING_DF = GOVERNMENT_SPENDING_DF.T
GOVERNMENT_SPENDING_DF.columns = GOVERNMENT_SPENDING_DF.iloc[0]
GOVERNMENT_SPENDING_DF = GOVERNMENT_SPENDING_DF.drop(["Country Name"])
GOVERNMENT_SPENDING_DF = GOVERNMENT_SPENDING_DF.apply(pd.to_numeric, errors='coerce')
GOVERNMENT_SPENDING_DF


# ## Government tax revenue by % GDP

# In[4]:


GOVERNMENT_TAXREVENUE_DF = pd.read_csv("govt_tax_revenue_by_gdp.csv")
GOVERNMENT_TAXREVENUE_DF = GOVERNMENT_TAXREVENUE_DF.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
GOVERNMENT_TAXREVENUE_DF = GOVERNMENT_TAXREVENUE_DF.T
GOVERNMENT_TAXREVENUE_DF.columns = GOVERNMENT_TAXREVENUE_DF.iloc[0]
GOVERNMENT_TAXREVENUE_DF = GOVERNMENT_TAXREVENUE_DF.drop(["Country Name"])
GOVERNMENT_TAXREVENUE_DF = GOVERNMENT_TAXREVENUE_DF.apply(pd.to_numeric, errors='coerce')
GOVERNMENT_TAXREVENUE_DF


# ## State owned enterprise by % GDP

# In[5]:


STATE_OWNED_ENTERPRISES_DF = pd.read_csv("state_owned_enterprise_gdp_converted.csv")
STATE_OWNED_ENTERPRISES_DF = STATE_OWNED_ENTERPRISES_DF.set_index("Year")
STATE_OWNED_ENTERPRISES_DF = STATE_OWNED_ENTERPRISES_DF.apply(pd.to_numeric, errors='coerce')
STATE_OWNED_ENTERPRISES_DF


# ## Government subsidies and transfers by % GDP

# In[6]:


GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = pd.read_csv("govt_subsidies_by_expense.csv")
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.T
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.columns = GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.iloc[0]
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.drop(["Country Name"])
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF

# Normalize by gdp
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.apply(pd.to_numeric, errors='coerce')
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF = (GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF/100) * (GOVERNMENT_SPENDING_DF/100) * 100
GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF


# ## GDP Growth Annum

# In[7]:


GDP_GROWTH_ANNUM_DF = pd.read_csv("gdp_growth_annum.csv")
GDP_GROWTH_ANNUM_DF = GDP_GROWTH_ANNUM_DF.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
GDP_GROWTH_ANNUM_DF = GDP_GROWTH_ANNUM_DF.T
GDP_GROWTH_ANNUM_DF.columns = GDP_GROWTH_ANNUM_DF.iloc[0]
GDP_GROWTH_ANNUM_DF = GDP_GROWTH_ANNUM_DF.drop(["Country Name"])
GDP_GROWTH_ANNUM_DF = GDP_GROWTH_ANNUM_DF.apply(pd.to_numeric, errors='coerce')
GDP_GROWTH_ANNUM_DF


# ## Developed Countries DF

# In[8]:


developed_countries = ['Austria', 'Belgium', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 
                       'Greece', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 
                       'Netherlands', 'Norway', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 
                       'Switzerland', 'United Kingdom', 'Cyprus', 'Hong Kong SAR, China', 'Israel', 'Japan', 'Macao SAR, China', 
                       'Singapore', 'Korea, Rep.', 'Canada', 'United States', 'Australia', 'New Zealand']

not_listed_govt_subsidies = []
not_listed_govt_consump = []
not_listed_state_owned = []
not_listed_govt_tax = []

for country in developed_countries:
    if country not in GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.columns:
        not_listed_govt_subsidies.append(country)
        
    if country not in GOVERNMENT_CONSUMPTION_DF.columns:
        not_listed_govt_consump.append(country)
        
    if country not in STATE_OWNED_ENTERPRISES_DF.columns:
        not_listed_state_owned.append(country)
        
    if country not in GOVERNMENT_TAXREVENUE_DF.columns:
        not_listed_govt_tax.append(country)


# In[9]:


print("in govt subsidies: " , not_listed_govt_subsidies)
print("in govt consump: " , not_listed_govt_consump)
print("in govt state owned: " , not_listed_state_owned)
print("in govt tax: " , not_listed_govt_tax)


# In[10]:


developed_countries_df_2015 = pd.DataFrame(columns=['country', 'govt_subsidies', 'govt_consumption', 'state_owned', 'tax_revenue'])

for country in developed_countries:
    developed_countries_df_2015 = developed_countries_df_2015.append({'country': country,
                                                                     'govt_subsidies': GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF[country]["2015"],
                                                                     'govt_consumption': GOVERNMENT_CONSUMPTION_DF[country]["2015"],
                                                                     'state_owned': STATE_OWNED_ENTERPRISES_DF[country]["2015"],
                                                                     'tax_revenue': GOVERNMENT_TAXREVENUE_DF[country]["2015"]}, ignore_index=True)


# In[11]:


developed_countries_df_2015 = developed_countries_df_2015.dropna()
developed_countries_df_2015 = developed_countries_df_2015.set_index("country")
developed_countries_df_2015


# ## Developing Countries

# In[12]:


developing_countries = ['Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 
                        'Aruba', 'Azerbaijan', 'Bahamas, The', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belize', 
                        'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 
                        'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Central African Republic', 'Chad', 
                        'Chile', 'China', 'Colombia', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', 
                        'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 
                        'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Fiji', 'Gabon', 
                        'Gambia, The', 'Georgia', 'Ghana', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 
                        'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iran, Islamic Rep.', 'Jamaica', 'Jordan', 
                        'Kazakhstan', 'Kenya', 'Kuwait', 'Kyrgyz Republic', 'Lao PDR', 'Lebanon', 'Lesotho', 
                        'Liberia', 'Libya', 'Macedonia, FYR', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 
                        'Mauritania', 'Mauritius', 'Mexico', 'Micronesia, Fed. Sts.', 'Moldova', 
                        'Mongolia', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 
                        'Nicaragua', 'Niger', 'Nigeria', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 
                        'Paraguay', 'Peru', 'Philippines', 'Poland', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 
                        'St. Kitts and Nevis', 'St. Lucia', 'St. Vincent and the Grenadines', 'Samoa', 'Sao Tome and Principe', 
                        'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Solomon Islands', 
                        'South Africa', 'South Sudan', 'Sri Lanka', 'Sudan', 'Suriname', 'Syrian Arab Republic', 'Tajikistan', 'Tanzania', 
                        'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
                        'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay',
                        'Vanuatu', 'Venezuela, RB', 'Vietnam', 'Yemen, Rep.', 'Zambia']

not_listed_developing_govt_subsidies = []
not_listed_developing_govt_consump = []
not_listed_developing_state_owned = []
not_listed_developing_govt_tax = []

for country in developing_countries:
    if country not in GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF.columns:
        not_listed_developing_govt_subsidies.append(country)
        
    if country not in GOVERNMENT_CONSUMPTION_DF.columns:
        not_listed_developing_govt_consump.append(country)
        
    if country not in STATE_OWNED_ENTERPRISES_DF.columns:
        not_listed_developing_state_owned.append(country)
        
    if country not in GOVERNMENT_TAXREVENUE_DF.columns:
        not_listed_developing_govt_tax.append(country)


# In[13]:


developing_countries_df_2015 = pd.DataFrame(columns=['country', 'govt_subsidies', 'govt_consumption', 'state_owned', 'tax_revenue'])

for country in developing_countries:
    developing_countries_df_2015 = developing_countries_df_2015.append({'country': country,
                                                                     'govt_subsidies': GOVERNMENT_SUBSIDIES_AND_TRANSFERS_DF[country]["2015"],
                                                                     'govt_consumption': GOVERNMENT_CONSUMPTION_DF[country]["2015"],
                                                                     'state_owned': STATE_OWNED_ENTERPRISES_DF[country]["2015"],
                                                                     'tax_revenue': GOVERNMENT_TAXREVENUE_DF[country]["2015"]}, ignore_index=True)


# In[14]:


developing_countries_df_2015 = developing_countries_df_2015.dropna()
developing_countries_df_2015 = developing_countries_df_2015.set_index("country")
developing_countries_df_2015


# # K-Means Clustering

# ### Developing Countries

# In[15]:


cluster = KMeans(n_clusters = 2, max_iter=2000)


# In[16]:


developing_countries_df_2015["category"] = cluster.fit_predict(developing_countries_df_2015.values)

developing_category1 = developing_countries_df_2015.loc[developing_countries_df_2015['category'] == 1]
developing_category0 = developing_countries_df_2015.loc[developing_countries_df_2015['category'] == 0]

plt.scatter(developing_category1["govt_subsidies"], developing_category1["govt_consumption"], c="green")
plt.scatter(developing_category0["govt_subsidies"], developing_category0["govt_consumption"], c="blue")
plt.figure(figsize=(20,10))


# In[17]:


developing_countries_df_2015


# ### Developed Countries

# In[30]:


developed_countries_df_2015["category"] = cluster.fit_predict(developed_countries_df_2015.values)

point1 = developed_countries_df_2015.loc[developed_countries_df_2015['category'] == 1]
point2 = developed_countries_df_2015.loc[developed_countries_df_2015['category'] == 0]

plt.scatter(point1["govt_subsidies"], point1["govt_consumption"], c="green")
plt.scatter(point2["govt_subsidies"], point2["govt_consumption"], c="blue")
plt.figure(figsize=(20,10))


# In[31]:


developed_countries_df_2015


# ## GDP Growth Annum Developed Countries

# In[48]:


developed_countries_df_2015["growth_rate"] = [GDP_GROWTH_ANNUM_DF[country]["2015"] for country in developed_countries if country in developed_countries_df_2015.index]

point1 = developed_countries_df_2015.loc[developed_countries_df_2015['category'] == 1]
point2 = developed_countries_df_2015.loc[developed_countries_df_2015['category'] == 0]

developed_countries_df_2015


# In[49]:


print("Low Government Intervention: \n" , point1.index, "\n")
print("High Government Intervention: \n", point2.index)


# In[50]:


growth_mean = developed_countries_df_2015["growth_rate"].mean()
low_intervention_growth_mean = point1["growth_rate"].mean()
high_intervention_growth_mean = point2["growth_rate"].mean()

print("growth mean: ",growth_mean)
print("low countries growth mean: ", low_intervention_growth_mean)
print("high countries growth mean: ", high_intervention_growth_mean)


# In[51]:


#FIND OUTLIERS

# calculate summary statistics
data_mean, data_std = np.mean(point1["growth_rate"]), np.std(point1["growth_rate"])

# identify outliers
cut_off = data_std * 2
lower, upper = data_mean - cut_off, data_mean + cut_off

outliers = [x for x in point1["growth_rate"] if x < lower or x > upper]

country = GDP_GROWTH_ANNUM_DF.T
country_name = {}

for outlier in outliers:
    country_name[country.loc[country['2015'] == outlier].index[0]] = outlier


print("Standard Deviation:", data_std)
print("Bollinger Bands   :", lower, upper)
print("Countries         :", country_name)


# In[52]:


#FIND OUTLIERS

# calculate summary statistics
data_mean, data_std = np.mean(point2["growth_rate"]), np.std(point2["growth_rate"])

# identify outliers
cut_off = data_std * 2
lower, upper = data_mean - cut_off, data_mean + cut_off

outliers = [x for x in point2["growth_rate"] if x < lower or x > upper]

country = GDP_GROWTH_ANNUM_DF.T
country_name = {}

for outlier in outliers:
    country_name[country.loc[country['2015'] == outlier].index[0]] = outlier


print("Standard Deviation:", data_std)
print("Bollinger Bands   :", lower, upper)
print("Countries         :", country_name)


# ## GDP Growth Annum Developing Countries

# In[53]:


developing_countries_df_2015["growth_rate"] = [GDP_GROWTH_ANNUM_DF[country]["2015"] for country in developing_countries if country in developing_countries_df_2015.index]

developing_category1 = developing_countries_df_2015.loc[developing_countries_df_2015['category'] == 1]
developing_category0 = developing_countries_df_2015.loc[developing_countries_df_2015['category'] == 0]

developing_countries_df_2015


# In[54]:


print("Low Government Intervention: \n" , developing_category0.index, "\n")
print("High Government Intervention: \n", developing_category1.index)


# In[55]:


growth_mean = developing_countries_df_2015["growth_rate"].mean()
low_intervention_growth_mean = developing_category0["growth_rate"].mean()
high_intervention_growth_mean = developing_category1["growth_rate"].mean()

print("growth mean: ",growth_mean)
print("low countries growth mean: ", low_intervention_growth_mean)
print("high countries growth mean: ", high_intervention_growth_mean)


# In[56]:


#FIND OUTLIERS

# calculate summary statistics
data_mean, data_std = np.mean(developing_category0["growth_rate"]), np.std(developing_category0["growth_rate"])

# identify outliers
cut_off = data_std * 2
lower, upper = data_mean - cut_off, data_mean + cut_off

outliers = [x for x in developing_category0["growth_rate"] if x < lower or x > upper]

country = GDP_GROWTH_ANNUM_DF.T
country_name = {}

for outlier in outliers:
    country_name[country.loc[country['2015'] == outlier].index[0]] = outlier


print("Standard Deviation:", data_std)
print("Bollinger Bands   :", lower, upper)
print("Countries         :", country_name)


# In[57]:


#FIND OUTLIERS

# calculate summary statistics
data_mean, data_std = np.mean(developing_category1["growth_rate"]), np.std(developing_category1["growth_rate"])

# identify outliers
cut_off = data_std * 2
lower, upper = data_mean - cut_off, data_mean + cut_off

outliers = [x for x in developing_category1["growth_rate"] if x < lower or x > upper]

country = GDP_GROWTH_ANNUM_DF.T
country_name = {}

for outlier in outliers:
    country_name[country.loc[country['2015'] == outlier].index[0]] = outlier


print("Standard Deviation:", data_std)
print("Bollinger Bands   :", lower, upper)
print("Countries         :", country_name)

