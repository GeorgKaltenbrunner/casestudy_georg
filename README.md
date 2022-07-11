# casestudy_georg

## Ziel:
Vorhersagen der Auswirkung von zwei Szenarien auf das Geschäft eines Unternehmens, das Fahrräder in Seoul vermietet.

## Vorgehensweise:
### 1. datensatz.py
#### Ziel:
- Erste Einblicke in Datensatz erhalten
- Ausreißer identfizieren
- Duplikate entfernen
- Fehlende Werte ersetzen
#### Vorgehensweise:
- Erste Einblicke (head(), tail(), describe() usw.)
- Univariate Analyse (Histogramm, Boxplots)
- Bivariate Analyse (Korrelationsmatrix)
#### Wichtig:
Datensatz wurde bereinigt:
- Werte für 2017 wurden entfernt
- Ausreißer wurden über Spalte 'Temperature(°C)' behandelt
- csv_nach_cleaning.csv wird hier exportiert
### 2. data_exploration.py
#### Ziel:
- Testen von zuvor entwickelten Hypothesen
- Besseres Verständnis von dem Datensatz erhalten
- Wichtige Zusammenhänge verstehen
### 3. predictive_modeling.py
#### Ziel:
- Entwickeln eines Vorhersagemodells für die Nachfrage nach Mieträdern unter Berücksichtigung der Szenarien
#### Modell:
- Multiple Regression
- from sklearn import linear_model
#### Vorgehensweise:
- Zur Ermittlung der Zusammensetzung der Variablen zunächst verschiedene Kombinationen getestet
- Test erfolgte durch Eingabe der Variablen und der Zielvariable ('Rented Bike Count')
- Performance getestet über Übergabe der Variablen-Werte und Berechnung des R2 anhand der 'predicted Rented Bike Count' und der tatsächlichen
#### Wichtig:
- Variablen für Modell: Temperature(°C)', 'Humidity(%)', 'Rainfall(mm)'
- predicted.csv wird hier exportiert
### 4. ergebnisse_visualisiern.py
#### Ziel:
- Visualisieren der Ergebnisse, durch Analyse und Vergleich von predicted.csv mit csv_nach_cleaning.csv
