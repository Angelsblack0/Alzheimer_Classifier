import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carica il dataset
df = pd.read_csv("alzheimers_disease_data.csv")
plt.figure(figsize=(7, 5))
df["Diagnosis"].value_counts().plot(kind="bar", color=["green", "blue"], edgecolor="black")
plt.xlabel("Diagnosi (0: Non malato 1: Malato)")
plt.ylabel("Conteggio")
plt.title("Distribuzione dei malati")
plt.xticks(rotation=0)
plt.show()