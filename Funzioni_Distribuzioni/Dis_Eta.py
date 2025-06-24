import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Crea l'istogramma dell'età (normalizzato in percentuale)
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=40, edgecolor="black", alpha=0.7, density=True)

# Aggiungi etichette per gli assi e il titolo
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.title("Age Distribution of Patients (Normalized)")

# Mostra il grafico
plt.show()
