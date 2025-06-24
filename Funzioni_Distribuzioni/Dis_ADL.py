import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Crea l'istogramma dei punteggi ADL
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="ADL", kde=True, hue="Diagnosis", stat="count", bins=100)

# Aggiungi etichette e titolo
plt.title("Distribuzione dei punteggi ADL nel dataset")
plt.xlabel("Punteggio ADL")
plt.ylabel("Numero di soggetti")

# Mostra il grafico
plt.show()

