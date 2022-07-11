#!/usr/bin/env python
# coding: utf-8

# ## Predictive Modeling
# 
# 1. Multiple Regression:
# 
#     a) zunächst nur mit Temperatur und Humidity
#     
#     b) Können weitere Variablen eingebaut werden?
#     
# 2. Einbindung der Eintrittswahrscheinlichkeiten
# - Ein DataFrame, wo beide Szenarien eingebunden sind

# In[260]:


# Imports

import pandas as pd
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[3]:


# Read Data

df_master = pd.read_csv(r'csv_nach_cleaning.csv')
df = df_master.copy()


# In[176]:


# Speichere einmal alle Werte für Temperatur und Humidity ab und die Zielwerte (Rented Bike Count)

X = df[['Temperature(°C)', 'Humidity(%)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Generiere 

predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['Temperature(°C)'][i], df['Humidity(%)'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rented Bike Count'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted'], color='blue', label='predicted')
plt.plot(predicted_grouped['observed'], color='red', label='observed')
plt.legend(loc='best')
plt.title('Predicted vs. observed')
plt.show()

# Checke Performance mean_squared_error

print('Mean Squarred: %f' % mean_squared_error(predicted_df['predicted'].values.tolist(), predicted_df['observed'].values.tolist()))

# Checke Performance R2

print('R2: %f ' % r2_score(predicted_df['observed'].values.tolist(),predicted_df['predicted'].values.tolist()))


# Ergebnis: 32% des Datensatzes ist von der Vorhersage abgedeckt

# In[180]:


# Speichere einmal alle Werte für Temperatur, Humidity, Rainfall ab und die Zielwerte (Rented Bike Count)

X = df[['Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)


predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['Temperature(°C)'][i], df['Humidity(%)'][i], df['Rainfall(mm)'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rented Bike Count'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted'], color='blue', label='predicted')
plt.plot(predicted_grouped['observed'], color='red', label='observed')
plt.legend(loc='best')
plt.title('Predicted vs. observed')
plt.show()

# Checke Performance mean_squared_error

print('Mean Squarred: %f' % mean_squared_error(predicted_df['predicted'].values.tolist(), predicted_df['observed'].values.tolist()))

# Checke Performance R2

print('R2: %f ' % r2_score(predicted_df['observed'].values.tolist(),predicted_df['predicted'].values.tolist()))


# Ergebnis: Performance nimmt zu

# In[181]:


# Speichere einmal alle Werte für Temperatur, Humidity, Dew point temperature(°C) ab und die Zielwerte (Rented Bike Count)

X = df[['Temperature(°C)', 'Humidity(%)', 'Dew point temperature(°C)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)


predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['Temperature(°C)'][i], df['Humidity(%)'][i], df['Dew point temperature(°C)'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rented Bike Count'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted'], color='blue', label='predicted')
plt.plot(predicted_grouped['observed'], color='red', label='observed')
plt.legend(loc='best')
plt.title('Predicted vs. observed')
plt.show()

# Checke Performance mean_squared_error

print('Mean Squarred: %f' % mean_squared_error(predicted_df['predicted'].values.tolist(), predicted_df['observed'].values.tolist()))

# Checke Performance R2

print('R2: %f ' % r2_score(predicted_df['observed'].values.tolist(),predicted_df['predicted'].values.tolist()))


# Ergebnis: Bei drei Variabeln hat Rainfall einen höheren Einfluss, als Dew point temperature(°C)

# In[173]:


# Speichere einmal alle Werte für Temperatur, Humidity, Rainfall, Dew point temperature(°C) ab und die Zielwerte (Rented Bike Count)

X = df[['Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)', 'Dew point temperature(°C)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Generiere 

predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['Temperature(°C)'][i], df['Humidity(%)'][i], df['Rainfall(mm)'][i], df['Dew point temperature(°C)'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rented Bike Count'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted'], color='blue', label='predicted')
plt.plot(predicted_grouped['observed'], color='red', label='observed')
plt.legend(loc='best')
plt.title('Predicted vs. observed')
plt.show()

# Checke Performance mean_squared_error

print('Mean Squarred: %f' % mean_squared_error(predicted_df['predicted'].values.tolist(), predicted_df['observed'].values.tolist()))

# Checke Performance R2

print('R2: %f ' % r2_score(predicted_df['observed'].values.tolist(),predicted_df['predicted'].values.tolist()))


# In[172]:


# Speichere einmal alle Werte für Temperatur, Humidity, Rainfall, Dew point temperature(°C), Visibility ab und die Zielwerte (Rented Bike Count)

X = df[['Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)', 'Dew point temperature(°C)', 'Visibility (10m)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Generiere 

predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['Temperature(°C)'][i], df['Humidity(%)'][i], df['Rainfall(mm)'][i], df['Dew point temperature(°C)'][i], df['Visibility (10m)'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rented Bike Count'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted'], color='blue', label='predicted')
plt.plot(predicted_grouped['observed'], color='red', label='observed')
plt.legend(loc='best')
plt.title('Predicted vs. observed')
plt.tight_layout()
plt.show()

# Checke Performance mean_squared_error

print('Mean Squarred: %f' % mean_squared_error(predicted_df['predicted'].values.tolist(), predicted_df['observed'].values.tolist()))

# Checke Performance R2

print('R2: %f ' % r2_score(predicted_df['observed'].values.tolist(),predicted_df['predicted'].values.tolist()))


# Ergebnis: bei fünf Variablen nimmt die Genauigkeit der Vorhersage im Vergleich zu drei Variablen(mit Rainfall) nicht signifikant zu
# 
# -> Für weiteres Vorgehen Rainfall mit einbeziehen

# ### Szenarien
# 
# 1. Werte aus den Zielwerten berechnen
# 2. Rainfall-Werte aus den neuen Temperature und Humidity Werten berechnen

# In[270]:


# 1. Werte aus den Zielwerten berechnen

# Annahme die Temperaturzunahme verteilt sich gleichmäßig über den gesamten Zeitraum
# Zur Verteilung der Eintrittswahrscheinlichkeiten wird eine Zufallszahl generiert und darüber folgende Regel:
# Zufallszahl < 8: Szenario Temperaturanstieg tritt ein, sonst Szenario Humidityanstieg

temp_generiert = []
humidity_generiert = []

for value in range(len(df)):
    random_value = random.randint(1,10)
    
    if random_value < 8:
        temp_generiert.append(df['Temperature(°C)'][value] + 2)
        humidity_generiert.append(df['Humidity(%)'][value])
    else:
        humidity_generiert.append(df['Humidity(%)'][value] + 3)
        temp_generiert.append(df['Temperature(°C)'][value])
        
df['temp_szenario'] = temp_generiert
df['humidity_szenario'] = humidity_generiert


# In[271]:


# 2. Vorhersage der neuen Rainfall Daten

X = df[['Temperature(°C)', 'Humidity(%)']]
Y = df['Rainfall(mm)']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Generiere angepasste Rainfall Werte

predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[df['temp_szenario'][i], df['humidity_szenario'][i]]])
    predicted_values.append(prediction.item())

predicted_df = pd.DataFrame({'predicted': predicted_values, 'observed': df['Rainfall(mm)'].values, 'date': df['Date']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').median()


# In[272]:


rainfall_szenario = predicted_values


# In[277]:


# Eigene DataFrame für die Szenarien Werte

szenario_df = pd.DataFrame({'date': df['Date'], 'temp_szenario': df['temp_szenario'], 'humidity_szenario': df['humidity_szenario'], 'rainfall_szenario': rainfall_szenario})


# In[290]:


# Vorhersage der Mietrad Nachfrage über die Variablen 'Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)'

X = df[['Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)']]
Y = df['Rented Bike Count']

# Initialisiere LinearRegression und übergebe Daten

regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Generiere eine vorhergesagte Mietrad Nachfrage

predicted_values = []

for i in range(len(df)):
    prediction = regr.predict([[szenario_df['temp_szenario'][i], szenario_df['humidity_szenario'][i], szenario_df['rainfall_szenario'][i]]])
    predicted_values.append(prediction.item())

# Speichere die vorhergesagten Werte in predicted_df ab 
predicted_df = pd.DataFrame({'predicted_bike_demand': predicted_values, 'date': szenario_df['date'], 'temperatur_szenario': szenario_df['temp_szenario'], 'humidity_szenario': szenario_df['humidity_szenario'], 'rainfall_szenario': szenario_df['rainfall_szenario']})
predicted_df['date'] = pd.to_datetime(predicted_df['date'])
predicted_grouped = predicted_df.groupby(by='date').mean()

# Plot
plt.figure(figsize=(15, 9))
plt.plot(predicted_grouped['predicted_bike_demand'], color='blue', label='predicted')
plt.legend(loc='best')
plt.title('Predicted Bike Count')
plt.tight_layout()
plt.show()


# In[287]:


# Exportiere predicted_df

predicted_df.to_csv('predicted.csv')

