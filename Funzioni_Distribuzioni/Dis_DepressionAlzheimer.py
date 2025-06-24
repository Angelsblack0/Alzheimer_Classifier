import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Filtra solo i soggetti depressi
depressi = df[df['Depression'] == 1]

# Conta quanti depressi sono malati e quanti no
malati = depressi[depressi['Diagnosis'] == 1].shape[0]
non_malati = depressi[depressi['Diagnosis'] == 0].shape[0]
totale_depressi = malati + non_malati

# Calcola le percentuali
percentuale_malati = (malati / totale_depressi) * 100
percentuale_non_malati = (non_malati / totale_depressi) * 100

# Stampa i valori
print(f"Totale depressi: {totale_depressi}")
print(f"Malati tra i depressi: {malati} ({percentuale_malati:.2f}%)")
print(f"Non malati tra i depressi: {non_malati} ({percentuale_non_malati:.2f}%)")

# Grafico a torta
plt.figure(figsize=(6, 6))
plt.pie([malati, non_malati],
        labels=['Malati', 'Non malati'],
        autopct='%1.1f%%',
        colors=['#ff6666', '#99ccff'],
        startangle=90)
plt.title("Distribuzione Alzheimer tra i pazienti depressi")
plt.axis('equal')
plt.show()
