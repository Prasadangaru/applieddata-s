import pandas as pd

def C02_emission(filename):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename, header=2)

    # Set the "Country Name" column as the index
    df.set_index('Country Name', inplace=True)

    # Select columns from "1960" to "2021"
    df = df.loc[:, '1960':'2021']

    # Transpose the DataFrame so that the countries are columns and the years are rows
    df = df.transpose()

    # Load the CSV file into a Pandas DataFrame
    df_countries = pd.read_csv(filename, skiprows=4)

    # Set the "Country Name" column as the index
    df_countries.set_index('Country Name', inplace=True)

    # Select columns from "1960" to "2021"
    df_countries = df_countries.loc[:, '1960':'2021']
    
    return df, df_countries


#to output years as rows and countries as columns
years,countries = C02_emission('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5355819.csv')

years

#to output countries as rows and years as columns
countries


countries.info()

years.info()

countries.describe()

years.info()



countries.corr()

years.corr()


#cleaning the dataframe 
import pandas as pd

# Read in the CSV file
df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_5358096.csv", skiprows=4)

# Drop columns that contain no data
df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace=True)

# Remove any rows that contain all NaN values
df.dropna(how="all", inplace=True)

# Rename the "Country Name" column to "Country"
df.rename(columns={"Country Name": "Country"}, inplace=True)

# Set the "Country" column as the index
df.set_index("Country", inplace=True)

# Remove any commas in the data and convert to numeric values
df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", ""), errors="coerce"))

# Print the cleaned dataframe
print(df)


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5355819.csv", skiprows=4)

# Select the required columns and rows
countries = ["China", "India", "Japan", "Russian Federation", "Germany"]
years = [str(i) for i in range(2010, 2016)]
data = df.loc[df["Country Name"].isin(countries), ["Country Name"] + years]

# Set Country Name as the index and transpose the dataframe
data = data.set_index("Country Name").T

# Plot the data as a multi-line graph
plt.plot(data.index.astype(int), data["China"], label="China")
plt.plot(data.index.astype(int), data["India"], label="India")
plt.plot(data.index.astype(int), data["Japan"], label="Japan")
plt.plot(data.index.astype(int), data["Russian Federation"], label="Russia")
plt.plot(data.index.astype(int), data["Germany"], label="Germany")

plt.xlabel("Year")
plt.ylabel("CO2 emissions (kt)")
plt.title("CO2 emissions (kt) from 2010 to 2015")
plt.legend()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv", skiprows=4)

# Select the required columns and rows
countries = ["China", "India", "Japan", "Russian Federation", "Germany"]
years = [str(i) for i in range(2010, 2016)]
data = df.loc[df["Country Name"].isin(countries), ["Country Name"] + years]

# Set Country Name as the index and transpose the dataframe
data = data.set_index("Country Name").T

# Plot the data as a multi-line graph
plt.plot(data.index.astype(int), data["China"], label="China")
plt.plot(data.index.astype(int), data["India"], label="India")
plt.plot(data.index.astype(int), data["Japan"], label="Japan")
plt.plot(data.index.astype(int), data["Russian Federation"], label="Russia")
plt.plot(data.index.astype(int), data["Germany"], label="Germany")

plt.xlabel("Year")
plt.ylabel("Electricity production from oil, gas and coal sources (% of total)")
plt.title("Electricity production from oil, gas and coal sources (% of total) from 2010 to 2015")
plt.legend()
plt.show()



#China's CO2 Emissions from 1960 to 2008
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5355819.csv", skiprows=4)

# Select the required data for China in the specified years
china_co2 = df.loc[df["Country Name"] == "China", ["Country Name", "1960", "1972", "1984", "1996", "2008"]]
china_co2 = china_co2.set_index("Country Name").T

# Create a heatmap using matplotlib
plt.imshow([china_co2.values], cmap='YlGnBu')
plt.xticks(range(china_co2.shape[0]), china_co2.index, rotation=45)
plt.yticks([])
plt.colorbar(label='CO2 Emissions (kt)')
plt.title("China's CO2 Emissions from 1960 to 2008")
plt.show()


#China's population from 1960 to 2008
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_5358096.csv", skiprows=4)

# Select the required data for China in the specified years
china_pop = df.loc[df["Country Name"] == "China", ["Country Name", "1960", "1972", "1984", "1996", "2008"]]
china_pop = china_pop.set_index("Country Name").T

# Create a heatmap using matplotlib
plt.imshow([china_pop.values], cmap="YlGnBu")
plt.xticks(range(len(china_pop)), china_pop.index)
plt.yticks([], [])
plt.xlabel("Year")
plt.title("China's population from 1960 to 2008")
plt.colorbar()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_5358096.csv", skiprows=4)

# Select the required data for the specified countries and years
countries = ["China", "India", "Russia", "Japan", "Germany"]
years = [str(year) for year in range(2010, 2016)]
data = df.loc[df["Country Name"].isin(countries), ["Country Name"] + years]
data = data.set_index("Country Name")

# Create a bar plot using matplotlib
ax = data.plot(kind="bar", figsize=(10, 6), width=0.8)
ax.set_xlabel("Country")
ax.set_ylabel("Population")
ax.set_title("Population of Selected Countries from 2010 to 2015")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5355819.csv", skiprows=4)

# Select the required data for the specified countries and years
countries = ["China", "India", "Russia", "Japan", "Germany"]
years = [str(year) for year in range(2010, 2016)]
co2_data = df.loc[df["Country Name"].isin(countries), ["Country Name"] + years]
co2_data = co2_data.set_index("Country Name")

# Create a bar plot using matplotlib
co2_data.plot(kind="bar")
plt.xticks(rotation=0)
plt.xlabel("Country")
plt.ylabel("C02 Emissions (kt)")
plt.title("C02 Emissions for Selected Countries from 2010 to 2015")
plt.show()

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv files
agricultural_land = pd.read_csv('API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5352105.csv', skiprows=4)
forest_area = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5355065.csv', skiprows=4)
electricity_production = pd.read_csv('API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv', skiprows=4)
co2_emissions = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5355819.csv', skiprows=4)
population_total = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_5358096.csv', skiprows=4)

# filter the data for China only
agricultural_land_china = agricultural_land[agricultural_land['Country Name'] == 'China']
forest_area_china = forest_area[forest_area['Country Name'] == 'China']
electricity_production_china = electricity_production[electricity_production['Country Name'] == 'China']
co2_emissions_china = co2_emissions[co2_emissions['Country Name'] == 'China']
population_total_china = population_total[population_total['Country Name'] == 'China']

# create a new dataframe with the necessary columns and rows
years = ['1960', '1980', '2000', '2020']
data = {'Agricultural land': [], 'Forest area': [], 'Electricity production from oil and gas': [], 'CO2 emissions': [], 'Population total': []}

for year in years:
    data['Agricultural land'].append(agricultural_land_china[year].iloc[0])
    data['Forest area'].append(forest_area_china[year].iloc[0])
    data['Electricity production from oil and gas'].append(electricity_production_china[year].iloc[0])
    data['CO2 emissions'].append(co2_emissions_china[year].iloc[0])
    data['Population total'].append(population_total_china[year].iloc[0])

df = pd.DataFrame(data, index=years)

# calculate the correlation between the variables for each year
correlation_matrix = df.corr()

# plot the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(correlation_matrix, cmap='coolwarm')

# set ticks and labels
ax.set_xticks(np.arange(len(data)))
ax.set_yticks(np.arange(len(data)))
ax.set_xticklabels(data.keys(), fontsize=12, rotation=45)
ax.set_yticklabels(data.keys(), fontsize=12)
ax.xaxis.tick_top()

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=12)

# add text annotations
for i in range(len(data)):
    for j in range(len(data)):
        text = ax.text(j, i, round(correlation_matrix.iloc[i, j], 2), ha='center', va='center', color='white', fontsize=14)

# set title
ax.set_title('Correlation Matrix for China', fontsize=16, pad=20)

# show the plot
plt.show()
