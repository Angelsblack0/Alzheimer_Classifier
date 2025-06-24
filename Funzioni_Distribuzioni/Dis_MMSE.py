import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Calcola la percentuale di malati (Diagnosis=1) in base ai punteggi continui di MMSE
percentuali_mmse_continuo = df.groupby("MMSE")["Diagnosis"].mean() * 100

# Ordina i valori per punteggio MMSE (nel caso non lo siano)
percentuali_mmse_continuo = percentuali_mmse_continuo.sort_index()

# Crea il grafico a barre (istogramma verticale)
plt.figure(figsize=(12, 6))
sns.barplot(x=percentuali_mmse_continuo.index, y=percentuali_mmse_continuo.values, color="skyblue")

# Aggiungi etichette e titolo
plt.title("Percentuale di soggetti con Alzheimer in base al punteggio MMSE")
plt.xlabel("Punteggio MMSE")
plt.ylabel("Percentuale di soggetti con Alzheimer")

# Opzionalmente mostra etichette su alcuni punti (es. ogni 5)
for i, (x, val) in enumerate(zip(percentuali_mmse_continuo.index, percentuali_mmse_continuo.values)):
    if i % 5 == 0:  # mostra ogni 5 per leggibilità
        plt.text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=8)

# Limita l'asse y
plt.ylim(0, 100)

# Mostra il grafico
plt.show()