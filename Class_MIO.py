import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Funzione che plotta
def valuta_modello(nome, modello, X_test, y_test, predizioni):
    print(f"\n--- {nome} ---")
    print("Parametri del modello:")
    
    # Parametri da escludere dalla stampa
    da_escludere = {
        'ccp_alpha', 'min_impurity_decrease', 'min_weight_fraction_leaf',
        'min_impurity_split', 'min_weight_fraction_leaf',
        'warm_start', 'validation_fraction', 'n_iter_no_change',
        'max_features', 'max_leaf_nodes', 'min_samples_split',
        'min_samples_leaf', 'splitter'
    }
    
    params = modello.get_params()
    params_filtrati = {k: v for k, v in params.items() if k not in da_escludere and v is not None}
    
    print(params_filtrati)

    print("\nClassification Report:")
    print(classification_report(y_test, predizioni))

    acc = accuracy_score(y_test, predizioni)
    prec = precision_score(y_test, predizioni, zero_division=0)
    rec = recall_score(y_test, predizioni, zero_division=0)
    f1 = f1_score(y_test, predizioni, zero_division=0)

    cm = confusion_matrix(y_test, predizioni)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"{nome}\nAccuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1-score: {f1:.2f}")
    plt.grid(False)
    plt.show()

# 1. Caricamento dati
df = pd.read_csv('Dataset/alzheimers_disease_data.csv')

# 2. Pulizia: togli colonne non numeriche (eccetto Diagnosis)
X = df.drop('Diagnosis', axis=1)
X = X.select_dtypes(include=[np.number])
y = df['Diagnosis']  # 0 = non malato, 1 = malato

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 4. Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dizionario per salvare f1-score
f1_scores = {}

# SVM
print("Stampa SVM")
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
valuta_modello("SVM", svc, X_test, y_test, pred_svc)
f1_scores["SVM"] = f1_score(y_test, pred_svc, zero_division=0)

# Random Forest
print("Stampa Random Forest")
rfc = RandomForestClassifier(n_estimators=300, random_state=42)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
valuta_modello("Random Forest", rfc, X_test, y_test, pred_rfc)
f1_scores["Random Forest"] = f1_score(y_test, pred_rfc, zero_division=0)

# Feature Importance (Random Forest)
importances = rfc.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importanza delle feature (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# Naive Bayes
print("Stampa Gauss")
nbc = GaussianNB()
nbc.fit(X_train, y_train)
pred_nbc = nbc.predict(X_test)
valuta_modello("Naive Bayes", nbc, X_test, y_test, pred_nbc)
f1_scores["Naive Bayes"] = f1_score(y_test, pred_nbc, zero_division=0)

# K-Nearest Neighbors
print("Stampa KNN")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
valuta_modello("K-Nearest Neighbors", knn, X_test, y_test, pred_knn)
f1_scores["KNN"] = f1_score(y_test, pred_knn, zero_division=0)

# Decision Tree
print("Stampa Decision Tree")
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
pred_dtc = dtc.predict(X_test)
valuta_modello("Decision Tree", dtc, X_test, y_test, pred_dtc)
f1_scores["Decision Tree"] = f1_score(y_test, pred_dtc, zero_division=0)

# Trova migliore e peggiore
migliore_nome = max(f1_scores, key=f1_scores.get)
peggiore_nome = min(f1_scores, key=f1_scores.get)

print(f"\nMigliore classificatore: {migliore_nome} (F1 = {f1_scores[migliore_nome]:.2f})")
print(f"Peggiore classificatore: {peggiore_nome} (F1 = {f1_scores[peggiore_nome]:.2f})")

# Modelli da testare per la GridSearch
modelli = {
    "SVM": (svc, {"C": list(range(1, 101)), "kernel": ['linear', 'rbf']}),  
    "Random Forest": (rfc, {"n_estimators": [50, 100, 200, 300], "max_depth": list(range(1, 100))}),
    "Naive Bayes": (nbc, {}),  # Nessun parametro per GaussianNB
    "KNN": (knn, {"n_neighbors": list(range(1, 101))}),
    "Decision Tree": (dtc, {"max_depth": list(range(2, 200, 2)), "criterion": ["gini", "entropy"]}),
}

# Ottimizzazione GridSearch per migliore
modello_migliore, param_migliore = modelli[migliore_nome]
grid_migliore = GridSearchCV(modello_migliore, param_migliore, scoring='f1', cv=10)
grid_migliore.fit(X_train, y_train)
pred_migliore = grid_migliore.predict(X_test)
valuta_modello(f"{migliore_nome} (Ottimizzato)", grid_migliore.best_estimator_, X_test, y_test, pred_migliore)
print("Migliori parametri trovati per", migliore_nome, ":", grid_migliore.best_params_)

# Ottimizzazione GridSearch per peggiore
modello_peggiore, param_peggiore = modelli[peggiore_nome]
grid_peggiore = GridSearchCV(modello_peggiore, param_peggiore, scoring='f1', cv=10)
grid_peggiore.fit(X_train, y_train)
pred_peggiore = grid_peggiore.predict(X_test)
valuta_modello(f"{peggiore_nome} (Ottimizzato)", grid_peggiore.best_estimator_, X_test, y_test, pred_peggiore)
print("Migliori parametri trovati per", peggiore_nome, ":", grid_peggiore.best_params_)
