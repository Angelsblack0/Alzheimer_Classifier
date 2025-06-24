import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Filtra solo chi ha in famiglia un parente con Alzheimer
famiglia_alz = df[df['FamilyHistoryAlzheimers'] == 1]

# Conta quanti di questi sono malati e quanti no
malati = famiglia_alz[famiglia_alz['Diagnosis'] == 1].shape[0]
non_malati = famiglia_alz[famiglia_alz['Diagnosis'] == 0].shape[0]
totale = malati + non_malati

# Calcola le percentuali
pct_malati = (malati / totale) * 100
pct_non_malati = (non_malati / totale) * 100

# Stampa i risultati
print(f"Totale con storia familiare: {totale}")
print(f"  Malati: {malati} ({pct_malati:.1f}%)")
print(f"  Non malati: {non_malati} ({pct_non_malati:.1f}%)")

# Grafico a torta
plt.figure(figsize=(6,6))
plt.pie(
    [malati, non_malati],
    labels=['Malati', 'Non malati'],
    autopct='%1.1f%%',
    colors=['#ff6666','#99ccff'],
    startangle=90
)
plt.title("Alzheimer tra chi ha storia familiare")
plt.axis('equal')
plt.show()
