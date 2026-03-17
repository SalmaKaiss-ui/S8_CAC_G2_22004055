# Projet Data Science : Classification des risques d'audit par cycle comptable

## 1. Contexte métier (Audit Financier & Contrôle Interne)

Dans le cadre légal et normatif de l'audit financier (particulièrement encadré par les normes ISA - International Standards on Auditing, et adapté par l'OECCA au Maroc), l'approche par les risques est la pierre angulaire de toute mission de commissariat aux comptes. L'objectif n'est pas de vérifier l'exhaustivité des transactions (ce qui est impossible), mais de concentrer les efforts d'audit sur les zones où le risque d'anomalies significatives est le plus élevé.

La classification préalable des cycles comptables selon leur niveau de risque (Faible, Modéré, Élevé) permet au responsable de mission de :
- Allouer de manière adéquate les ressources (ex: affecter un auditeur senior sur les cycles complexes).
- Définir l'étendue des tests de détails ("Substantive testing").
- Orienter la stratégie d'audit.

**Exemples concrets de risques par cycle :**
- **Cycle Achats / Fournisseurs :** Risque de fraude par création de fournisseurs fictifs, comptabilisation de factures fictives, non-respect de la séparation des tâches (ségrégation).
- **Cycle Ventes / Clients :** Reconnaissance anticipée du chiffre d'affaires (Cut-off), surévaluation des créances, provisionnement insuffisant des créances douteuses.
- **Cycle Paie / Personnel :** Maintien d'employés fictifs dans le système, heures supplémentaires non justifiées.
- **Cycle Trésorerie :** Détournements de fonds, fraudes au président, rapprochements bancaires non effectués ou fictifs (c'est souvent le cycle le plus sensible).
- **Cycle Stocks :** Obsolescence non provisionnée, surévaluation des quantités (ex: lors de l'inventaire physique).

---

## 2. Génération de données simulées

Afin de pouvoir entraîner nos modèles prédictifs, nous allons simuler un dataset réaliste (n > 1000). Ce dataset intègre les caractéristiques généralement observées lors d'une évaluation de contrôle interne.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Pour la modélisation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import shap

# Fixer la graine pour reproductibilité
np.random.seed(42)
n_samples = 1500

# 1. Variables Catégorielles
cycles = ['Achats', 'Ventes', 'Paie', 'Trésorerie', 'Stocks']
cycle_data = np.random.choice(cycles, n_samples)
digitalisation_data = np.random.choice(['Faible', 'Moyenne', 'Elevée'], n_samples, p=[0.2, 0.5, 0.3])

# 2. Variables Numériques (Features)
volume_transactions = np.abs(np.random.normal(50000, 20000, n_samples)).astype(int)
complexite_si = np.random.randint(1, 6, n_samples) # Score de 1 (simple) à 5 (complexe)
historique_anomalies = np.random.poisson(1.5, n_samples) # Nombre d'anomalies détectées N-1
ratio_financier = np.abs(np.random.normal(12, 5, n_samples)) # Ex: Rotation, DSO...
score_controle_interne = np.round(np.random.uniform(30, 100, n_samples), 2) # Note sur 100

# Création du DataFrame
df = pd.DataFrame({
    'cycle': cycle_data,
    'volume_transactions': volume_transactions,
    'complexite_si': complexite_si,
    'historique_anomalies': historique_anomalies,
    'ratio_financier': ratio_financier,
    'score_controle_interne': score_controle_interne,
    'niveau_digitalisation': digitalisation_data
})

# 3. Création d'une variable cible (Risque) basée sur des règles métier bruitées
# Création d'un score de risque "caché"
risk_score = (
    df['historique_anomalies'] * 15 +
    df['complexite_si'] * 10 +
    (100 - df['score_controle_interne']) * 0.8 +
    np.random.normal(0, 10, n_samples) # Ajout de bruit pour réalisme
)

# Ajustement selon le niveau de digitalisation
risk_score = np.where(df['niveau_digitalisation'] == 'Faible', risk_score + 15, risk_score)
risk_score = np.where(df['niveau_digitalisation'] == 'Elevée', risk_score - 10, risk_score)

# Ajustement par cycle (Trésorerie généralement plus risquée par nature)
df['risk_score_temp'] = risk_score
df.loc[df['cycle'] == 'Trésorerie', 'risk_score_temp'] += 15
df.loc[df['cycle'] == 'Achats', 'risk_score_temp'] += 5

# Classification 
conditions = [
    (df['risk_score_temp'] < 55),
    (df['risk_score_temp'] >= 55) & (df['risk_score_temp'] < 85),
    (df['risk_score_temp'] >= 85)
]
choices = ['Faible', 'Modéré', 'Élevé']
df['risque_audit'] = np.select(conditions, choices, default='Modéré')
df.drop('risk_score_temp', axis=1, inplace=True)

print(df.head())
print("\nDistribution de la variable cible :")
print(df['risque_audit'].value_counts(normalize=True)*100)
```

---

## 3. Préparation des données & Feature Engineering

La préparation des données inclut l'encodage des variables catégorielles, la mise à l'échelle des variables numériques et un **Feature Engineering** orienté métier.

```python
# Feature Engineering : Création d'indicateurs composites
# Par exemple, un ratio volume / contrôle interne
df['indice_vulnerabilite'] = df['volume_transactions'] / (df['score_controle_interne'] + 1)
df['alerte_historique'] = (df['historique_anomalies'] > 2).astype(int)

# Séparation Features et Target
X = df.drop('risque_audit', axis=1)
y = df['risque_audit']

# Encodage de la variable cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Cartographie inversée : {0: 'Faible', 1: 'Modéré', 2: 'Élevé'} (Attention l'ordre dépend de l'alphabet, vérifions via le.classes_)
classes_namemap = dict(zip(le.transform(le.classes_), le.classes_))

# Séparation Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Pipeline de preprocessing
numeric_features = ['volume_transactions', 'complexite_si', 'historique_anomalies', 
                    'ratio_financier', 'score_controle_interne', 'indice_vulnerabilite', 'alerte_historique']
categorical_features = ['cycle', 'niveau_digitalisation']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

---

## 4. Modélisation : Comparaison des algorithmes

Nous allons utiliser le `Pipeline` de scikit-learn pour évaluer plusieurs algorithmes : **Régression Logistique, Random Forest, SVM, KNN et XGBoost**.

```python
# Définition des modèles à comparer
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = {}

for name, model in models.items():
    # Création du pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Entraînement
    clf.fit(X_train, y_train)
    
    # Prédiction
    y_pred = clf.predict(X_test)
    
    # Évaluation de base
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'Pipeline': clf, 'Accuracy': acc, 'Predictions': y_pred}
    print(f"Modèle : {name} | Accuracy : {acc:.4f}")

# Recherche des hyperparamètres (GridSearchCV) sur le meilleur modèle pressenti (ex: Random Forest)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"\nMeilleur Random Forest Accuracy (GridSearch) : {accuracy_score(y_test, best_rf.predict(X_test)):.4f}")
```

---

## 5. Évaluation des performances

Prenons le meilleur modèle sélectionné via XGBoost ou Random Forest et analysons les résultats plus en détail.

```python
best_model_name = 'XGBoost' # Supposons que XGBoost ou RF est le meilleur
best_pipeline = results[best_model_name]['Pipeline']
y_pred_best = results[best_model_name]['Predictions']

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.ylabel('Vraies Classes')
plt.xlabel('Classes Prédites')
plt.show()
```

---

## 6. Visualisation & Explicabilité (SHAP et ROC)

L'interprétabilité est fondamentale en audit. Un auditeur refusera une boîte noire.

```python
# 6.1 Importance des variables (Feature Importance globale)
# On extrait le modèle et le preprocessor du Pipeline
rf_model = grid_search.best_estimator_.named_steps['classifier']
feature_names = numeric_features + list(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importance des variables (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align="center", color='darkblue')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.show()

# 6.2 SHAP Values (Pour l'interprétabilité locale)
# Transformation des données pour XGBoost/RF
X_test_transformed = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)
xgb_model = results['XGBoost']['Pipeline'].named_steps['classifier']

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# Résumé des effets de chaque variable
shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", class_names=le.classes_)
```

---

## 7. Interprétation Métier

La sortie des visualisations nous permet de tirer des conclusions totalement en phase avec les intuitions des normes d'audit :

1. **Le rôle critique du Contrôle Interne :**
   Le `score_controle_interne` est généralement la composante qui possède le poids (feature importance) le plus lourd. En audit, un environnement de contrôle interne fort permet logiquement de baisser le niveau de risque perçu.

2. **L'historique des anomalies :**
   L'indicateur `historique_anomalies` joue un rôle de pénalité ou d'alerte. Les SHAP values prouvent que plus l’historique est chargé, plus le modèle tend vers la classe "Élevé". Cela reflète la norme ISA 315.

3. **Vulnérabilité des cycles de Trésorerie & Achats :**
   Les variables catégorielles (Cycle = Trésorerie) orientent le modèle vers des probabilités de risque élevé de manière mécanique. Le chiffre d'affaires (Ventes) est aussi observé, mais la fraude aux décaissements pénalise sévèrement les Achats et la Trésorerie.

---

## 8. Cas d'Usage Concret en Cabinet d'Audit

**Comment ce modèle est-il utilisé lors d'une mission de CAC (Commissariat aux Comptes) ?**

- **Phase de Planification (Interim) :**
  L'auditeur récupère le Fichier des Écritures Comptables (FEC), remplit le questionnaire de contrôle interne, et évalue la complexité du SI du client (ou s'appuie sur le rapport des auditeurs IT). 
- **Prédiction :**
  Il injecte ces paramètres dans le modèle. Le modèle classe instantanément les cycles.
- **Allocation des ressources (Stratégie d'Audit) :**
  Si le modèle sort : *Achats (Risque Élevé)*, *Trésorerie (Risque Élevé)*, *Personnel (Risque Faible)*.
  L'associé signataire va décider d'allouer un manager senior sur la vérification des rapprochements bancaires et faire appel à des Data Analysts pour mener des tests de validation massifs sur les décaissements fournisseurs. Le cycle personnel ne subira que des examens analytiques de base.

---

## 9. Limites et axes d'améliorations

**Limites actuelles :**
- **Données simulées :** Bien que basées sur des heuristiques métier réelles, ces données ne captent pas la granularité complexe de réels environnements frauduleux. L'adaptation à une entreprise réelle peut s'avérer complexe sans historique de données labellisées précis.
- **Biais subjectif :** Le score de contrôle interne reste dépendant de l'appréciation subjective de l'auditeur qui a rempli le questionnaire initial.

**Améliorations futures :**
- **Machine Learning non supervisé (Isolation Forest) :** Au lieu de classer les cycles, on pourrait classer *chaque transaction* comptable du FEC pour détecter l'anomalie individuelle (au niveau de l'écriture de journal).
- **Intégration du NLP :** Analyser automatiquement par Deep Learning (BERT) les PV de conseils d'administration pour en extraire l'indicateur "Ton on the top" (le climat éthique de la direction) et l'intégrer aux *features*.
- **Export :** Utiliser `joblib` ou `mlflow` pour packager le modèle dans une API, qu'une interface web métier pourrait interroger directement lors de la phase de planification.
