import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Rimuovi le colonne che non vuoi includere nel confronto
df = df.drop(columns=['PatientID', 'Ethnicity'])

# Calcola la matrice di correlazione
correlazioni = df.corr(numeric_only=True)

# Mostra la mappa di calore con numeri
plt.figure(figsize=(16, 12))
sns.heatmap(correlazioni, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            annot_kws={'size': 5}, square=True, cbar_kws={'shrink': 0.8},
            xticklabels=correlazioni.columns, yticklabels=correlazioni.columns)
plt.title("Matrice di Correlazione tra le variabili")
plt.show()
