import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Filtra solo i soggetti con disorientamento
disorientati = df[df["Disorientation"] == 1]

# Conta quanti di questi sono malati e quanti no
malati = disorientati[disorientati["Diagnosis"] == 1].shape[0]
non_malati = disorientati[disorientati["Diagnosis"] == 0].shape[0]
totale = malati + non_malati

# Calcola percentuali
pct_malati = (malati / totale) * 100
pct_non_malati = (non_malati / totale) * 100

# Stampa i risultati
print(f"Totale con disorientamento: {totale}")
print(f"  Malati: {malati} ({pct_malati:.1f}%)")
print(f"  Non malati: {non_malati} ({pct_non_malati:.1f}%)")

# Grafico a torta
plt.figure(figsize=(6,6))
plt.pie(
    [malati, non_malati],
    labels=['Malati', 'Non malati'],
    autopct='%1.1f%%',
    colors=['#ffcc99','#66b3ff'],
    startangle=90
)
plt.title("Alzheimer tra chi ha Disorientamento")
plt.axis('equal')
plt.show()
