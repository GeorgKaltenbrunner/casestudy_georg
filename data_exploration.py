#!/usr/bin/env python
# coding: utf-8

# ## Data exploration
# 
# Hier wurden 10 Hypothesen erarbeitet, die im Folgenden getestet werden
# 
# - H1: Die meisten Räder werden nicht in den Ferien gemietet -> RICHTIG
# - H2: Je mehr es regnet, desto weniger wird gemietet -> RICHTIG
# - H3: Je höher die Temperatur, desto weniger wird gemietet -> Zwischen 8 und 28 Grad Celsius werden die meisten Bikes gemietet
# - H4: Je höher die Temperatur, desto höher ist humidity -> FALSCH
# - H5: Die Anzahl der gemieteten Bikes variiert zwischen den Jahreszeiten -> Im Winter am Geringsten
# - H6: Je stärker der Wind, desto weniger wird gemietet -> RICHTIG
# - H7: Die Windstärke hängt mit der humidity zusammen -> wenn die humidity zunimmt geht die Wind Geschwindigkeit zurück
# - H8: Die Anzahl der gemieteten Bikes hängt vom Monat ab -> der Rückgang der Nachfrage von Bikes pro Monat zwischen Juli und September ist zunächst auf den zunehmenden Regen und anschließend auf die Holidays zurückzuführen
# - H9: Es gibt einen Schwellenwert (Temperatur), ab welchen die Anzahl der gemieteten Bikes abnimmt -> siehe DBSCAN
# - H10: Die Nachfrage variiert zwischen den Uhrzeiten
# 

# In[1]:


# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# In[2]:


# Read Data

df_master = pd.read_csv(r'csv_nach_cleaning.csv')
df = df_master.copy()


# In[3]:


df.info()


# In[4]:


# Change Date Dtype to DateTime Format

df['Date'] = pd.to_datetime(df['Date'])
df.info()


# In[5]:


# H1: Die meisten Räder werden nicht in den Ferien gemietet

display(df.groupby(by="Holiday").sum())
df['Holiday'].value_counts()


# Ergebnis H1: Die meisten Räder werden nicht in den Ferien gemietet. 
# Daraus ergiebt sich, dass die Zielgruppe des Unternehmens nicht Touristen sind.
# 
# Frage: Werden mehr Räder unter der Woche gemietet? Daraus könnte sich ergeben, dass es Personen sind, die das Rad im Alltag benötigen

# In[6]:


# Füge Wochentag hinzu

df['day_of_week'] = df['Date'].dt.day_name()
df_dayoftheweek = df.loc[df['Holiday'] == 'No Holiday']
df_dayoftheweek['day_of_week'].value_counts()


# Ergebnis: Am Sonntag wurden die meisten Räder gemietet, Freitags die wenigsten.
# Frage: an welchen Tagen ist die Mietzeit am Größten?

# In[7]:


df.groupby(by="day_of_week").mean()


# Ergebnis: Spalte Hour ist die Uhrzeit

# In[8]:


# H2: Je mehr es regnet, desto weniger wird gemietet

anzahl_miete_regen = df.groupby(by="Rainfall(mm)")['Rented Bike Count'].sum()

anzahl_miete_regen = pd.DataFrame(anzahl_miete_regen)


# In[9]:


df.groupby(by="Rainfall(mm)")['Rented Bike Count'].sum()


# In[10]:


# Gruppiere Regen

regen_1 = df.loc[df['Rainfall(mm)'] <= 1.0]
regen_5 = df.loc[(df['Rainfall(mm)'] > 1.0) & (df['Rainfall(mm)'] <= 5.0)]
regen_10 = df.loc[(df['Rainfall(mm)'] > 5.0) & (df['Rainfall(mm)'] <= 10.0)]
regen_rest = df.loc[(df['Rainfall(mm)'] > 10.0)]

regen_dict = {'1mm': regen_1['Rented Bike Count'].sum(), '5mm': regen_5['Rented Bike Count'].sum(), '10mm': regen_10['Rented Bike Count'].sum(), '>10mm': regen_rest['Rented Bike Count'].sum()}

regen_df = pd.DataFrame(regen_dict, index=[0]).transpose()
#regen_df.rename(columns={'0': 'observed'}, inplace=True)
regen_df['Ratio'] = regen_df[0]/regen_df[0].sum()
regen_df


# In[11]:


# Werte für Mieträder je Rainfall in X abspeichern

X = df[['Rented Bike Count','Rainfall(mm)']].values

# Get nearest neighbours

neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(X) # fitting the data to the object
distances,indices=nbrs.kneighbors(X)

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show()


# In[12]:


# DBSCAN

dbscan = DBSCAN(eps = 10, min_samples = 4).fit(X) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot DBSCAN

plt.figure(figsize=(15, 9))
plt.scatter(X[:, 1], X[:,0], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Temperatur") # X-axis label
plt.ylabel("Bookings") # Y-axis label
plt.show() # showing the plot


# Ergebnis: Da 99% der gemieten Bikes bis 1mm Regenfall waren, wird angenommen, dass je weniger es regnet, die Anzahl der gemieten Bikes höher ist

# In[13]:


# H3: Je höher die Temperatur, desto weniger wird gemietet

temp = df.groupby(by="Temperature(°C)")['Rented Bike Count'].sum()
temp.plot()


# In[14]:


# Gruppiere die Temperatur und zähle die Anzahl der Mieträder

# Sichere in temp_list die Intervallbereiche, beginnend mit dem kleinsten Temperaturwert
temp_list = []
start_value = temp.index.min()
temp_list.append(start_value)
new_value = 0

# Set interval
interval = 10

# So lange, wie der new_value kleiner als der größte Temperaturwert ist
while new_value < temp.index.max():
    new_value = temp_list[-1] + interval
    temp_list.append(new_value)


# In[15]:


# Runden

temp_list = [round(num) for num in temp_list]
temp_list


# In[16]:


# Füge zu den Intervallen die Anzahl der Mieträder hinzu und das Ratio

temp_dict = dict()
for i in range(len(temp_list)):
    
    # Für den ersten Fall
    if i == 0:
        value = df.loc[df['Temperature(°C)'] <= temp_list[i]]
        temp_dict[temp_list[i]] = value['Rented Bike Count'].sum()
        
    # Für den letzten Fall
    elif i == len(temp_list)-1:
        value = df.loc[(df['Temperature(°C)'] > temp_list[i])]
        temp_dict[temp_list[i]] = value['Rented Bike Count'].sum()
    
    # Für die restlichen Bedingungen
    else:
        value = df.loc[(df['Temperature(°C)'] > temp_list[i]) & (df['Temperature(°C)'] <= temp_list[i+1])]
        temp_dict[temp_list[i]] = value['Rented Bike Count'].sum()
        
temp_df = pd.DataFrame(temp_dict, index=[0]).transpose()
#regen_df.rename(columns={'0': 'observed'}, inplace=True)
temp_df['Ratio'] = temp_df[0]/temp_df[0].sum()
display(temp_df)
temp_df = temp_df.sort_index()
temp_df['Ratio'].plot()


# Tabelle gelesen:
# Bsp: größer als -2 bis einschließlich 8 Grad Celsius hat das Ratio 0.35
# 
# Ergebnis: Zwischen den Temperaturen 8 und 28 Grad Celsius werden die meisten Bikes gemietet

# In[17]:


# H4: Je höher die Temperatur, desto höher ist humidity

humidity = df[['Temperature(°C)', 'Humidity(%)']]


# In[18]:


humidity = humidity.groupby(by="Temperature(°C)").sum()


# In[19]:


# Gruppiere die Temperatur und füge die Humidity hinzu

# Sichere in temp_list die Intervallbereiche, beginnend mit dem kleinsten Temperaturwert

temp_humidity_list = []
start_value = humidity.index.min()
temp_humidity_list.append(start_value)
new_value = 0

# Set interval
interval = 10

while new_value < humidity.index.max():
    new_value = temp_humidity_list[-1] + interval
    temp_humidity_list.append(new_value)

temp_humidity_list = [round(num) for num in temp_humidity_list]
temp_humidity_list


# In[20]:


# Füge zu den Intervallen die Humidity hinzu und das Ratio

temp_humidity_dict = dict()
for i in range(len(temp_humidity_list)):
    
    # Für den ersten Fall
    if i == 0:
        value = df.loc[df['Temperature(°C)'] <= temp_humidity_list[i]]
        temp_humidity_dict[temp_humidity_list[i]] = value['Humidity(%)'].mean()
        
    # Für den letzten Fall
    elif i == len(temp_humidity_list)-1:
        value = df.loc[(df['Temperature(°C)'] > temp_humidity_list[i])]
        temp_humidity_dict[temp_humidity_list[i]] = value['Humidity(%)'].mean()
    
    # Für die restlichen Bedingungen
    else:
        value = df.loc[(df['Temperature(°C)'] > temp_humidity_list[i]) & (df['Temperature(°C)'] <= temp_humidity_list[i+1])]
        temp_humidity_dict[temp_humidity_list[i]] = value['Humidity(%)'].mean()
        
temp_humidity_df = pd.DataFrame(temp_humidity_dict, index=[0]).transpose()
#regen_df.rename(columns={'0': 'observed'}, inplace=True)
temp_humidity_df['Ratio'] = temp_humidity_df[0]/temp_humidity_df[0].sum()
display(temp_humidity_df)
temp_humidity_df['Ratio'].plot()


# Ergebnis: ab -2 Grad Celsius bleibt die durchscnittliche Humidity gleich, trotz steigender Temperatur
# 
# NaN bei 38: Mittelwert wird berechnet, da es keine Werte gibt, die größer als 38 Grad Celsius haben, NaN, da durch 0 geteilt

# In[21]:


# H5: Die Anzahl der gemieteten Bikes variiert zwischen den Jahreszeiten

season = df.groupby(by="Seasons").sum()
season['Ratio'] = season['Rented Bike Count']/season['Rented Bike Count'].sum()
season


# Ergebnis: Im Winter ist die Nachfrage am geringsten

# In[22]:


# H6: Je stärker der Wind, desto weniger wird gemietet

wind = df.groupby(by="Wind speed (m/s)").sum()
wind['Rented Bike Count'].plot()


# In[23]:


wind['Rented Bike Count'].idxmax()


# Nach Abgleich mit (https://www.wetterkontor.de/de/bft_tabelle.html) ist bis 5.5 m/s Windstärke ein leichter Wind, der vmtl. dennoch spürbar als Gegenwind wird.
# 
# Ergebnis: Bis 1.2 m/s nimmt die Anzahl der gemieteten Bikes zu. Hier kann angenommen, werden, dass der Wind so minimal ist, dass er keinen Einfluss hat. Danach nimmt die Nachfrage mit zunehmender Windgeschwindigkeit stetig ab

# In[24]:


# H7: Die Windstärke hängt mit der humidity zusammen

humidity_wind = df[['Humidity(%)', 'Wind speed (m/s)']]


# In[25]:


humidity_wind.groupby(by="Humidity(%)").mean().plot()


# In[26]:


humidity_wind.corr(method ='pearson')


# Ergebnis: wenn die humidity zunimmt geht die Wind Geschwindigkeit zurück

# In[27]:


# H8: Die Anzahl der gemieteten Fahrrädern hängt vom Monat ab

bikes_month = df[['month', 'Rented Bike Count']]


# In[28]:


bikes_month.corr(method ='pearson')


# In[29]:


bikes_month.groupby(by="month").sum().plot()


# In[30]:


# Besten Monate hinsichtlich der gemieteten Fahrräder über Ratio ermittelt

bikes = bikes_month.groupby(by="month").sum()
bikes['ratio'] = bikes['Rented Bike Count']/bikes['Rented Bike Count'].sum()
bikes['ratio'].sort_values(ascending=False)


# Ergebnis: 
# - niedrigste Nachfrage in den Monaten Januar, Februar, Dezember
# - höchste Nachfrage in den Monate Juli, Juni, Oktober
# 
# Interessant: Von Juli bis September geht die Nachfrage zurück. 
# -> Hypothese: Temperatur nimmt zu, deshalb geht die Nachfrage zurück

# In[31]:


# Durchschnittliche Temperatur pro Monat

temp_month = df[['month','Temperature(°C)']]
temp_month.groupby(by="month").mean().plot()


# Ergebnis: Temperatur geht ebenfalls zurück. Rückgang in der Nachfrage liegt also nicht an Temperatur
# 
# Hypothese: zwischen Juli und September sind holidays

# In[35]:


# Anzahl von Holidays pro Monat
holidays_month = df[['month', 'Holiday']]
holidays_month_holiday = holidays_month.loc[holidays_month['Holiday']=='Holiday']
holidays_month_holiday.groupby(by="month").count().plot()


# Ergebnis: Rückgang der gemieteten Bikes ist zum Teil durch die Ferien zu erklären
# 
# Hypothese: Regen nimmt zu

# In[36]:


# Regen pro Monat

regen_month = df[['month', 'Rainfall(mm)']]
regen_month.groupby(by="month").mean().plot()


# Ergebnis: von Juli zu August nimmt der Regen stark zu.
# 
# Ergebnis gesamt: der Rückgang der Nachfrage von Bikes pro Monat zwischen Juli und September ist zunächst auf den zunehmenden Regen und anschließend auf die Holidays zurückzuführen

# In[37]:


# H9: Es gibt einen Schwellenwert (Temperatur), ab welchen die Anzahl der gemieteten Bikes abnimmt

temp_schwellenwert = temp = df.groupby(by="Temperature(°C)")['Rented Bike Count'].mean()
temp_schwellenwert.plot()


# Frage: Warum ist die durschnittliche Anzahl von gemieteten Bikes bei Temperature > 30 Grad Celsius so hoch?

# In[38]:


temp_analyse = df[["Temperature(°C)", 'Rented Bike Count']]


# In[39]:


# Verstehe Ausreißer bei über 30 Grad Celsius

temp_analyse[temp_analyse["Temperature(°C)"] >= 30]['Rented Bike Count'].mean()


# In[40]:


# Anzahl der Mieträder bei Temperaturen über 30 Grad Celsius

t = temp_analyse[temp_analyse["Temperature(°C)"] >= 30]

t['Temperature(°C)'].value_counts()


# In[41]:


# Anzahl von Mieträdern zwischen 10 und 30 Grad Celsius

q = temp_analyse[(temp_analyse["Temperature(°C)"] < 30) & (temp_analyse["Temperature(°C)"] > 10)]


# In[42]:


v = q['Temperature(°C)'].value_counts().sort_values()


# In[43]:


v.to_frame().sort_index().plot()


# Zwischenergebnis: Höhere Werte sind darauf zurückzuführen, dass es weniger Messwerte bei der hohen Grad Celsius Temperatur gibt
# 
# Lösungs-Idee: Median verwenden

# In[44]:


# Gemietete Räder nach Temperatur median

temp_schwellenwert_median = df.groupby(by="Temperature(°C)")['Rented Bike Count'].median()
temp_schwellenwert_median.plot()


# Zwischenergebnis: Gleiches Bild wie bei Mittelwert
# 
# Lösungsidee: Wieder summe verwenden

# In[45]:


# Gemietete Räder nach Temperatur sum

temp_schwellenwert_sum = df.groupby(by="Temperature(°C)")['Rented Bike Count'].sum()
temp_schwellenwert_sum.plot()


# In[46]:


temp_schwellenwert_sum = temp_schwellenwert_sum.to_frame()


# In[47]:


type(temp_schwellenwert_sum)


# In[48]:


# Neue Spalte mit Temperatur für DBSCAN
temp_schwellenwert_sum['temperatur'] = temp_schwellenwert_sum.index.values


# In[49]:


temp_schwellenwert_sum


# In[50]:


# Werte für den DBSCAN

X = temp_schwellenwert_sum.values


# In[51]:


# Get nearest neighbours

neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(X) # fitting the data to the object
distances,indices=nbrs.kneighbors(X)

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show()


# Zwischenergebnis: bis ca 190 

# In[52]:


# DBSCAN

dbscan = DBSCAN(eps = 150, min_samples = 4).fit(X) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot DBSCAN

plt.figure(figsize=(15, 9))
plt.scatter(X[:, 1], X[:,0], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Temperatur") # X-axis label
plt.ylabel("Bookings") # Y-axis label
plt.show() # showing the plot


# In[53]:


bike_schwellenwert = temp_schwellenwert_sum.groupby(by=["Rented Bike Count"]).sum().sort_values(by=['Rented Bike Count', 'temperatur'], ascending=False)


# In[54]:


bike_schwellenwert.head()


# Ergebnis: Schwellenwert nicht offensichtlich

# In[57]:


# H10: Die Nachfrage variiert zwischen den Uhrzeiten
df.groupby(by=['Hour'])['Rented Bike Count'].sum().plot()
df.groupby(by=['Hour'])['Rented Bike Count'].sum().sort_values(ascending=False)


# Ergebnis: Nachfrage abends (ab 18 Uhr) und frühs (ab 8 Uhr) am größten
# 
# Frage: variiert die Nachfrage nach Uhrzeit zwischen den Monaten?

# In[56]:


months = df['month'].unique()

for month in months:
    df_hour_month = df.loc[df['month'] == month]
    hour_demand = df_hour_month.groupby(by=['Hour'])['Rented Bike Count'].sum()
    plt.plot(hour_demand)
    plt.title(month)
    plt.show()


# Ergebnis: Nachfrage nach Uhrzeit bleibt realtiv gleich

# In[ ]:




