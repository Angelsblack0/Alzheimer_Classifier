import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Filtra solo i soggetti che hanno 'BehavioralProblems' uguale a 1
con_problematiche_comportamentali = df[df['BehavioralProblems'] == 1]

# Conta quanti di questi sono malati e quanti no
malati = con_problematiche_comportamentali[con_problematiche_comportamentali['Diagnosis'] == 1].shape[0]
non_malati = con_problematiche_comportamentali[con_problematiche_comportamentali['Diagnosis'] == 0].shape[0]
totale_con_problematiche_comportamentali = malati + non_malati

# Calcola le percentuali
percentuale_malati = (malati / totale_con_problematiche_comportamentali) * 100
percentuale_non_malati = (non_malati / totale_con_problematiche_comportamentali) * 100

# Stampa i valori
print(f"Totale con problematiche comportamentali: {totale_con_problematiche_comportamentali}")
print(f"Malati tra i soggetti con problematiche comportamentali: {malati} ({percentuale_malati:.2f}%)")
print(f"Non malati tra i soggetti con problematiche comportamentali: {non_malati} ({percentuale_non_malati:.2f}%)")

# Grafico a torta
plt.figure(figsize=(6, 6))
plt.pie([malati, non_malati],
        labels=['Malati', 'Non malati'],
        autopct='%1.1f%%',
        colors=['#ff6666', '#99ccff'],
        startangle=90)
plt.title("Distribuzione Alzheimer tra i soggetti con problematiche comportamentali")
plt.axis('equal')
plt.show()
