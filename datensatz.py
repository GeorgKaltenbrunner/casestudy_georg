#!/usr/bin/env python
# coding: utf-8

# ### Explorative Analyse
# - Erste Einblicke
# - Univariate Analyse
# - Bivariate Analyse
# 

# In[1]:


# Imports

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats # for qq-plot
from scipy.stats import shapiro # Shapiro-Test
from scipy.stats import chisquare # Chisquare-Test
from scipy.stats import kstest # Kolmogorov-Smirnov 
import seaborn as sns # Korrelations-Matrix


# In[2]:


# Read Data

df_master = pd.read_csv(r'SeoulBikeData.csv', encoding= 'unicode_escape')
df = df_master.copy()


# ### Explorative Analyse
# #### Erste Einblicke
# - head() + tail()
# - info()
# - describe()
# - Normalverteilung
# - Duplikate
# - Ausreißer
# - Fehlende Werte
# 
# #### Univariate Analyse
# - Histogramm
# - Boxplot
# 
# #### Bivaraite Analyse
# - Korrelations-Matrix

# In[3]:


# head() + tail()

display(df.head())
display(df.tail())


# In[4]:


df['Hour'].value_counts()


# In[5]:


# info()

df.info()


# In[6]:


# Date Dtype zu DateTime

df['Date'] = pd.to_datetime(df['Date'])


# In[7]:


# describe()

df.describe(percentiles=[0.05, 0.9,0.95])


# In[8]:


# Normalverteilung

def check_normaldistribution(data, sig_niveau, name):
    
    try:
        print('column %s' % name)
        # Graphs
        # QQ-Plots
        stats.probplot(data, dist='norm', plot = plt)
        print('QQ-Plots')
        plt.title(name)
        plt.show()
        print('----------\n')
        
        # BoxPlots
        print('BoxPlots')
        plt.boxplot(data)
        plt.title(name)
        plt.show()
        print('----------\n')

        # Histogram
        print('Histogram')
        _ = plt.hist(data, bins='auto')
        plt.title(name)
        plt.show()
        print('----------\n')

        # Tests
        # Chi Square Test
        print('Chi-Square Test for %s' % name)
        stat, p = chisquare(data)
        print('stat=%.3f, p=%.3f\n' % (stat, p))
        if p > sig_niveau:
            print('Normal distributed')
        else:
            print('Not normal distributed')
        print('-----------\n')

        # Kolmogorov-Smirnov Test
        print('Kolmogorov-Smirnov Test for %s' % name)
        stat, p = kstest(data, 'norm')
        print('stat=%.3f, p=%.3f\n' % (stat, p))
        if p > sig_niveau:
            print('Normal distributed')
        else:
            print('Not normal distributed')
        print('-----------\n')
    
    except:
        print('Did not work for %s\n' % column)


sig_niveau = 0.05
for column in df.columns:
    check_normaldistribution(df[column], sig_niveau, column)


# In[9]:


# Duplikate

df.duplicated().unique()


# In[10]:


# Ausreißer

def find_outliners(data, name):
    
    try:
        print(name)
        plt.boxplot(data)
        plt.show()    
        print('Min rate:', min(data))
        print('Max rate:', max(data))
        print('Mean rate:', data.mean())
        print('Median rate:', data.median())
        print('++++++++\n')
    
    except:
        print(name,'\n')
    
for column in df.columns:
        find_outliners(df[column], column)


# Ausreißer sind in mehrern Spalten vorhanden
# 
# Mögliche Gründe?
# - Fehler in der Messung
# - Fehler in der Datenübertragung
# - Messgerät könnte beschädigt gewesen sein
# - Evtl. direkte Sonneneinstrahlug bei Messung
# 

# In[11]:


# Deep Dive in Spalte Temperatur, auch wenn keine Ausreißer erkennbar
# Nur Spalten Date und Temperatur
df_temperatur = df[['Date', 'Temperature(°C)']].copy()


# Spalte Temperatur als Index setzen
df_temperatur = df_temperatur.set_index('Date')

# Für 2017 weniger Werte, deshalb zum abgelich Werte ab 2018
df_temperatur[2018:].plot()


# Hier sind sehr große Temperatur Sprünge ereknnbar.
# Ist es normal, dass es in Seoul bspw. zwischen Januar und Februar einen Temperatur Unteschied von fast 40 Grad gibt?
# 
# Nach Abgleich mit (https://de.weatherspark.com/h/y/142033/2018/Historisches-Wetter-w%C3%A4hrend-des-Jahres-2018-in-Seoul-S%C3%BCdkorea#Figures-Temperature) werden mögliche Messfehler deutlich
# 
# 

# In[18]:


df.info()
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month


# In[19]:


df['year'].value_counts()


# In[20]:


# Deep Dive in Spalte Rainfall
# Nur Spalten Date und Rainfall
df_regen = df[['Date', 'Rainfall(mm)']].copy()

# Spalte Temperatur als Index setzen
df_regen = df_regen.set_index('Date')

# Für 2017 weniger Werte, deshalb zum Abgleich Werte ab 2018
df_regen[2018:].plot()


# Nach Abgleich mit (https://www.laenderdaten.info/Asien/Suedkorea/Klima.php) sind auch hier Ausreißer deutlich erkennbar

# ##### Ausreißer Zusammenfassung
# 
# Ausreißer sind deutlich erkennbar. Nach Abgleich mit historischen Daten aus dem www sind mögliche Messfehler in den Daten erkennbar.
# 
# Für das weitere Vorgehen werden Ausreißer entfernt.
# 
# Spalte, die miteinbezogen werden:
# - Temperature(°C)
# 
# Wichtig hierbei zunächst auf pro Monat zu filtern, damit nicht ein maximaler Wert festgelegt wird, sondern monatsabhängig stattfindet. Zudem Datensatz für 2018 weiterverwenden, da für 2017 sehr wenig Daten vorhanden sind (nichtmal 10%)

# In[21]:


# Filter den Datensatz auf 2018

df_2018 = df.loc[df['year'] == 2018]

df_2018.info()


# In[22]:


# Entferne Ausreißer
# Speichere nicht Ausreißer in df_ausreißer

df_ausreißer = pd.DataFrame()

quantile_high = 0.8
quantile_low = 0.1

months_unique = df_2018['month'].unique()

for month in months_unique:
    print('Monat:', month)
    
    x = df_2018.loc[df_2018['month']== month ]
    q_high = x["Temperature(°C)"].quantile(quantile_high)
    q_low = x["Temperature(°C)"].quantile(quantile_low)
    y = x.loc[x["Temperature(°C)"] < q_high]
    y = y.loc[y["Temperature(°C)"] > q_low]
    print(y)
    df_ausreißer = df_ausreißer.append(y)
    


# In[23]:


df_ausreißer = df_ausreißer.set_index('Date')


# In[24]:


df_ausreißer['Temperature(°C)'].plot()


# In[25]:


df_ausreißer['Rainfall(mm)'].plot()


# In[26]:


# Fehlende Werte

df_ausreißer.isna().sum()


# ### Univariate Analyse

# In[27]:


# Histogramm nach Ausreißer entfernen

def histogram(data, name):
    
    try:
        print(name)
        plt.hist(data, bins='auto')
        plt.title(name)
        plt.show()    
        
    
    except:
        print(name,'\n')
    
for column in df_ausreißer.columns:
        histogram(df_ausreißer[column], column)
    


# In[28]:


# Boxplot nach Ausreißer entfernen
# Ausreißer

def boxplot(data, name):
    
    try:
        print(name)
        plt.boxplot(data)
        plt.title(name)
        plt.show()    
        
    except:
        print(name,'\n')
    
for column in df_ausreißer.columns:
        boxplot(df_ausreißer[column], column)


# ### Bivariate Analyse
# 

# In[29]:


# Korrelationsmatrix

# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df_ausreißer.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[30]:


# Export df_ausreißer als csv

df_ausreißer.to_csv('csv_nach_cleaning.csv')


# In[ ]:





# In[ ]:




