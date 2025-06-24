import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Filtra solo i soggetti con MemoryComplaints = 1
con_memory_complaints = df[df["MemoryComplaints"] == 1]

# Conta quanti di questi sono malati e quanti no
malati = con_memory_complaints[con_memory_complaints["Diagnosis"] == 1].shape[0]
non_malati = con_memory_complaints[con_memory_complaints["Diagnosis"] == 0].shape[0]
totale = malati + non_malati

# Calcola le percentuali
pct_malati = (malati / totale) * 100
pct_non_malati = (non_malati / totale) * 100

# Stampa i risultati
print(f"Totale con problemi di memoria: {totale}")
print(f"  Malati: {malati} ({pct_malati:.1f}%)")
print(f"  Non malati: {non_malati} ({pct_non_malati:.1f}%)")

# Crea un grafico a torta
plt.figure(figsize=(6, 6))
plt.pie(
    [malati, non_malati],
    labels=['Malati', 'Non malati'],
    autopct='%1.1f%%',
    colors=['#ff9999', '#99ccff'],
    startangle=90
)
plt.title("Alzheimer tra chi ha problemi di memoria")
plt.axis('equal')
plt.show()
