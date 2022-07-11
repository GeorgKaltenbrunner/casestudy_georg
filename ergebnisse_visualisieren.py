#!/usr/bin/env python
# coding: utf-8

# In[123]:


# Imports

import pandas as pd
import matplotlib.pyplot as plt


# In[126]:


# Read Data

df_master_predicted = pd.read_csv('predicted.csv')
df_pred = df_master_predicted.copy()
df_master_original = pd.read_csv(r'csv_nach_cleaning.csv')
df_original = df_master_original.copy()


# In[127]:


# Lösche Spalte

df_pred.drop("Unnamed: 0", axis=1, inplace=True)

# Ersetzte negative Werte in df_pred['predicted_bike_demand'] mit 0 und humidity_szenario > 100 mit 100

df_pred.loc[(df_pred.predicted_bike_demand < 0), 'predicted_bike_demand'] = 0
df_pred.loc[(df_pred.humidity_szenario > 100), 'humidity_szenario'] = 100


# In[128]:


# Manipuliere df_pred
# Ändere date Dtype zu DateTime
df_pred['date'] = pd.to_datetime(df_pred['date'])
df_pred.info()

# Füge Monat zu df_pred hinzu
df_pred['month'] = pd.DatetimeIndex(df_pred['date']).month


# In[129]:


df_original.info()


# In[130]:


# Vorhergesagte Nachfrage nach Mieträdern gruppiert nach Monat

df_pred.groupby(by="month")['predicted_bike_demand'].sum().plot()


# In[131]:


# Zu- oder Abnahme der Nachfrage nach Mieträdern

df_pred['predicted_bike_demand'].sum() - df_original['Rented Bike Count'].sum()


# Ergebnis: Nachfrage nimmt zu

# In[171]:


print('Nachfrage_vorhergesagt: %f' % df_pred['predicted_bike_demand'].sum())
print('Nachfrage_2018: %f' % df_original['Rented Bike Count'].sum())


# In[153]:


# Plot für durchschnittliche Nachfrage Veränderung pro Monat

plt.figure(figsize=(15, 9))
plt.plot(df_pred.groupby(by="month")['predicted_bike_demand'].mean(), color='blue', label='Predicted_Mit_Szenarien')
plt.plot(df_original.groupby(by='month')['Rented Bike Count'].mean(), color='red', label='Rented_Bike_Count_2018')
plt.legend(loc='best')
plt.title('Durchschnittliche Nachfrage Pro Monat_Predicted vs. 2018')
plt.xlabel('Monat')
plt.ylabel('Anzahl_Durchschnitt')
plt.show()


# In[133]:


# Plot für Temperatur predicted vs. 2018

plt.figure(figsize=(15, 9))
plt.plot(df_pred.groupby(by="month")['temperatur_szenario'].mean(), color='green', label='Temperatur_Szenario')
plt.plot(df_original.groupby(by='month')['Temperature(°C)'].mean(), color='yellow', label='Temperatur_2018')
plt.legend(loc='best')
plt.title('Durchschnittliche Temperatur pro Monat_Predicted vs. 2018')
plt.xlabel('Monat')
plt.ylabel('Temperature(°C)')
plt.show()


# In[134]:


# Plot für Humidity predicted vs. 2018

plt.figure(figsize=(15, 9))
plt.plot(df_pred.groupby(by="month")['humidity_szenario'].mean(), color='green', label='Humidity_Szenario')
plt.plot(df_original.groupby(by='month')['Humidity(%)'].mean(), color='yellow', label='Humidity_2018')
plt.legend(loc='best')
plt.title('Durchschnittliche Humidity pro Monat_Predicted vs. 2018')
plt.xlabel('Monat')
plt.ylabel('Humidity(%)')
plt.show()


# In[135]:


# Plot für Rainfall predicted vs. 2018

plt.figure(figsize=(15, 9))
plt.plot(df_pred.groupby(by="month")['rainfall_szenario'].sum(), color='green', label='Rainfall_Szenario')
plt.plot(df_original.groupby(by='month')['Rainfall(mm)'].sum(), color='yellow', label='Rainfall_2018')
plt.legend(loc='best')
plt.title('Durchschnittlicher Rainfall pro Monat_Predicted vs. 2018')
plt.xlabel('Monat')
plt.ylabel('Rainfall(mm)')
plt.show()


# In[136]:


df_pred['rainfall_szenario'].sum() - df_original['Rainfall(mm)'].sum()


# In[157]:


# Analysiere df_pred

pred_bike_demand = df_pred.groupby(by="month")['predicted_bike_demand'].sum()
pred_bike_demand = pred_bike_demand.to_frame()

pred_bike_demand['ratio'] = pred_bike_demand['predicted_bike_demand'] / pred_bike_demand['predicted_bike_demand'].sum()


# In[158]:


# Nachfrage nach Monaten

pred_bike_demand.sort_values(by="ratio", ascending=False)


# In[138]:


df_pred.describe()


# In[164]:


# Füge Nachfrage von 2018 hinzu

df_pred['demand_original'] = df_original['Rented Bike Count']


# In[165]:


# Ermittle Differenz der Nachfrage

df_pred['Differenz'] = df_pred['predicted_bike_demand'] - df_pred['demand_original']


# In[168]:


df_pred.groupby(by='month').mean()


# In[177]:


df_original.groupby(by=["month",'Hour'])['Rented Bike Count'].sum().plot()


# In[ ]:




