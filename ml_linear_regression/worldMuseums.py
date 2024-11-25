from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import pycountry
import requests
import bs4
import re


def clean_world_cities():
    cities = pd.read_csv('worldcitiespop.csv')
    cities = cities.dropna(subset=['Population'])
    cities = cities.drop(columns=['AccentCity'])
    cities['City'] = cities['City'].str.lower()

    specific_country_names = { # pycountry doesn't give the right name for these countries
        'ru': 'Russia',
        'tw': 'Taiwan',
        'kr': 'South Korea',
        'tr': 'Turkey'
    }


    def get_country_name(code):
        if code in specific_country_names:
            return specific_country_names[code]
        try:
            return pycountry.countries.get(alpha_2=code).name
        except:
            return code
        
    cities['Country'] = cities['Country'].apply(get_country_name)

    cities = cities[(cities['City'] != 'washington') | (cities['Region'] == 'DC')] 
    cities = cities.drop(columns=['Region'])

    return cities


def get_museum_data():
    url = "https://en.wikipedia.org/wiki/List_of_most-visited_museums"
    try:
        response = requests.get(url)
        if not response.status_code == 200:
            return "No information found"
        
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        museums_table = soup.find('table',{'class':"wikitable"})
        df = pd.read_html(str(museums_table))
        df = pd.DataFrame(df[0])

        df['Visitors in 2023 and 2022'] = df['Visitors in 2023 and 2022'].apply(lambda x: re.sub(r"\([^()]*\)|\[[^\[\]]*\]", '', x))
        df['Location'] = df['Location'].str.replace('Washington, D.C., United States', 'Washington, United States', regex=False) 
        
        df[['City', 'Country']] = df['Location'].str.extract(r'^(.*?),\s*(.*?)$')
        df['City'].fillna(df['Location'], inplace=True)
        df['Country'].fillna(df['Location'], inplace=True)
        df['City'] = df['City'].str.lower().str.replace(r'\bcity\b', '', regex=True).str.strip()


        def standardize_visitors(visitor_str):
            clean_str = re.sub(r'[^\d.]', '', visitor_str)
            if 'million' in visitor_str.lower():
                return float(clean_str) * 1000000
            else:
                return float(clean_str)

        df['Visitors in 2023 and 2022'] = df['Visitors in 2023 and 2022'].apply(standardize_visitors)        
        df = df[df['Visitors in 2023 and 2022'] > 2000000] 

        return df

    except Exception as e:
        print("Error occurred:", e)
        return None
    
    
def scatter_plot(df, labels=False):
    plt.figure(figsize=(10, 7))
    plt.scatter(df['Population'], df['Visitors in 2023 and 2022'], color='blue')

    if labels:
        for i, name in enumerate(df['Name']):
            plt.annotate(name, (df['Population'][i], df['Visitors in 2023 and 2022'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.title('Museum visitors vs Population')
    plt.xlabel('Population')
    plt.ylabel('Museum visitors')
    plt.show()


def world_map(df):
    fig = px.scatter_geo(df, lat='Latitude', lon='Longitude', size='Visitors in 2023 and 2022', hover_name='Name', color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth", title="Museum visitors")
    fig.update_layout(geo=dict(landcolor="LightGreen", showocean=True, oceancolor="LightBlue"))
    fig.show()


def pie_chart_visitors(df, ax):
    grouped_df = df.groupby('Country')['Visitors in 2023 and 2022'].sum().reset_index()
    colors = [
        '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0','#ffb3e6', 
        '#c4e17f', '#f9bc86', '#a3e1d4', '#fb83fa', '#ffb367', '#fbed96'
    ]
    wedges, texts, autotexts = ax.pie(grouped_df['Visitors in 2023 and 2022'], autopct='%1.1f%%', colors=colors) 
    ax.legend(wedges, grouped_df['Country'], title='Countries', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1)) 
    ax.set_title('Visitors by country')


def pie_chart_population(df, ax):
    grouped_df = df.groupby('Country')['Population'].first().reset_index()
    colors = [
        '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0','#ffb3e6', 
        '#c4e17f', '#f9bc86', '#a3e1d4', '#fb83fa', '#ffb367', '#fbed96'
    ]
    wedges, texts, autotexts = ax.pie(grouped_df['Population'], autopct='%1.1f%%', colors=colors)
    ax.legend(wedges, grouped_df['Country'], title='Countries', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('Population by country')


def show_pie_charts(df):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.5)  
    pie_chart_visitors(df, ax[0])
    pie_chart_population(df, ax[1])
    plt.show()


def bar_chart_museums(df):
    museum_names = df['Name']
    visitor_counts = df['Visitors in 2023 and 2022']
    country_populations = df['Population']
    
    x_pos = np.arange(len(museum_names))  
    width = 0.35 

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_pos - width/2, visitor_counts, width, label='Visitors')
    ax.bar(x_pos + width/2, country_populations, width, label='Population')

    ax.set_xlabel('Museums')
    ax.set_ylabel('Counts')
    ax.set_title('Visitors and population by museum')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(museum_names, rotation=90)
    ax.legend()

    plt.show()


def linear_regression_model(df):
    X = df[['Population']]  # independent
    y = df['Visitors in 2023 and 2022']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print('Coefficients: ', model.coef_)
    print('Mean squared error: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))

    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Population')
    plt.ylabel('Museum visitors')
    plt.title('Linear regression: Population vs museum visitors')
    plt.legend()
    plt.show()


def linear_regression_model_log(df):
    # no linear relationship between population and visitors -> log to see if it's better 
    df['Log_Population'] = np.log(df['Population'])
    df['Log_Visitors'] = np.log(df['Visitors in 2023 and 2022'])

    X = df[['Log_Population']]
    y = df['Log_Visitors']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print('Coefficients: ', model.coef_)
    print('Mean squared error: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))

    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.xlabel('Log of population')
    plt.ylabel('Log of museum visitors')
    plt.title('Linear regression: Log of population vs log of museum visitors')
    plt.show()
   
   
def polynomial_regression_model(df):
    X = df[['Population']].values 
    y = df['Visitors in 2023 and 2022'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    polynominal_features = PolynomialFeatures(degree = 2)
    X_train_poly = polynominal_features.fit_transform(X_train)
    X_test_poly = polynominal_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    
    print('Coefficients: ', model.coef_)
    print('Mean squared error: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))

    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_fit_poly = polynominal_features.transform(X_fit)

    y_fit = model.predict(X_fit_poly)

    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Testing data')
    plt.plot(X_fit, y_fit, color='red', label=f'Polynomial degree 2')
    plt.xlabel('Population')
    plt.ylabel('Museum visitors')
    plt.title('Polynomial regression (degree2)')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    cities = clean_world_cities()
    museum_data = get_museum_data()
    merged_data = pd.merge(museum_data, cities, on=['City', 'Country'], how='inner') 

    scatter_plot(merged_data)
    scatter_plot(merged_data, True)

    world_map(merged_data)
    merged_data = merged_data.drop(columns=['Latitude', 'Longitude'])

    show_pie_charts(merged_data)

    bar_chart_museums(merged_data)

    linear_regression_model(merged_data)

    # remove Louvre (outlier)
    new_data = merged_data[merged_data['Name'] != 'Louvre']
    linear_regression_model(new_data)
    new_data = new_data[new_data['Name'] != "Musée d'Orsay"]
    linear_regression_model(new_data)

    linear_regression_model_log(merged_data)

    polynomial_regression_model(merged_data)


### Code improvements
# 1. Add more error handling to the functions
# 2. No population data for some cities (Dongguan, Hong Kong, Beijing, Vatican City) -> external data sources or approximate
# 3. Cross-validation for the models
# 4. Alternative models for the data (decision tree, random forest, ...)
# 5. Code depends on the data being available in the same format -> make it more flexible
# 6. Use more efficient methods (Pandas, Numpy, Scikit-learn) for data manipulation
# 7. Influential variables (GDP, tourism, reputation, ...) could be added to the model (not considered here)
# 8. More visualizations (pair plots, correlation matrix, ...)
