# ENCG Settat — Projet Data Science S8 · Classification des Risques d'Audit par ML

**Université Hassan 1er — École Nationale de Commerce et de Gestion de Settat**















---

## COMPTE RENDU DE PROJET DATA SCIENCE

### Classification des Risques d'Audit par Cycle Comptable
**Approche Machine Learning Supervisé**

| Intitulé | Classification ML des Risques d'Audit |
| :--- | :--- |
| **Module** | Data Science & Intelligence Artificielle |
| **Filière** | CAC — Contrôle, Audit & Comptabilité (L3 S8) |
| **Réalisé par** | Kaiss · Salma · Hamdoune · Oumaima |
| **Encadrant** |  Larhlimi Abderrahim |
| **Période** | Semestre 8 — Année Universitaire 2024–2025 |

**Année Universitaire 2024–2025**

---

## SOMMAIRE

1. [Introduction](#i-introduction)
    1. [Contexte général et thématique globale](#i1-contexte-général-et-thématique-globale)
    2. [Sujet choisi et problématique](#i2-sujet-choisi-et-problématique)
    3. [Méthodologie adoptée](#i3-méthodologie-adoptée)
2. [Revue de Littérature](#ii-revue-de-littérature)
    1. [L'audit financier et l'approche par les risques](#ii1-laudit-financier-et-lapproche-par-les-risques)
    2. [Machine Learning appliqué à l'audit](#ii2-machine-learning-appliqué-audit)
    3. [Interprétabilité des modèles](#ii3-interprétabilité-des-modèles)
3. [Présentation du Dataset](#iii-présentation-du-dataset)
    1. [Source et mode de génération](#iii1-source-et-mode-de-génération)
    2. [Description des variables](#iii2-description-des-variables)
    3. [Analyse exploratoire (EDA)](#iii3-analyse-exploratoire-eda)
4. [Développement — Modélisation et Pipeline](#iv-développement--modélisation-et-pipeline)
    1. [Préparation des données et Feature Engineering](#iv1-préparation-des-données-et-feature-engineering)
    2. [Algorithmes de classification comparés](#iv2-algorithmes-de-classification-comparés)
    3. [Optimisation et sélection du meilleur modèle](#iv3-optimisation-et-sélection-du-meilleur-modèle)
5. [Résultats et Discussion](#v-résultats-et-discussion)
    1. [Performances comparatives des modèles](#v1-performances-comparatives-des-modèles)
    2. [Matrice de confusion — Interprétation](#v2-matrice-de-confusion--interprétation)
    3. [Importance des variables](#v3-importance-des-variables)
    4. [Interprétation métier des résultats](#v4-interprétation-métier-des-résultats)
6. [Limites et Axes d'Amélioration](#vi-limites-et-axes-damélioration)
7. [Conclusion](#vii-conclusion)
8. [Bibliographie](#bibliographie)
9. [Annexes](#annexes)

---

## I. Introduction

### I.1 Contexte général et thématique globale
À l'ère de la transformation numérique, les professions comptables et d'audit font face à une révolution méthodologique sans précédent. L'intelligence artificielle et le Machine Learning s'imposent progressivement comme des outils complémentaires à la pratique traditionnelle de l'audit financier. La thématique globale de ce projet s'inscrit à l'intersection de deux domaines : la science des données (Data Science) et le contrôle-audit financier.

En contexte marocain, l'audit légal est encadré par les normes de l'OECCA qui s'alignent sur les normes ISA (International Standards on Auditing) émises par l'IAASB. Ces normes imposent une approche fondée sur les risques : l'auditeur doit concentrer ses diligences sur les zones présentant le plus fort potentiel d'anomalies significatives.

### I.2 Sujet choisi et problématique
Le sujet de ce projet porte sur la classification automatique des cycles comptables selon leur niveau de risque d'audit (Faible, Modéré, Élevé) à l'aide d'algorithmes de Machine Learning supervisé. La problématique centrale peut être formulée ainsi :

> Dans quelle mesure des algorithmes de Machine Learning supervisé peuvent-ils reproduire et améliorer le jugement professionnel d'un auditeur dans l'évaluation des risques inhérents aux cycles comptables ?

Les cinq cycles comptables étudiés sont ceux reconnus comme les plus critiques : Achats/Fournisseurs, Ventes/Clients, Paie/Personnel, Trésorerie et Stocks.

### I.3 Méthodologie adoptée
La démarche suit le processus CRISP-DM (Cross-Industry Standard Process for Data Mining) adapté au contexte d'audit :

1.  **Compréhension métier** — Étude des normes ISA et des facteurs de risque reconnus
2.  **Génération et exploration des données** — Dataset de 1 200 observations simulées
3.  **Préparation** — Encodage, normalisation, feature engineering orienté audit
4.  **Modélisation** — Comparaison de quatre algorithmes de classification supervisée
5.  **Évaluation** — Métriques de performance, matrice de confusion, validation croisée
6.  **Explicabilité** — Importance des variables pour valider auprès des praticiens

**Outils utilisés :** Python 3.x, scikit-learn, pandas, numpy, matplotlib, seaborn.

---

## II. Revue de Littérature

### II.1 L'audit financier et l'approche par les risques
L'audit financier est régi par un corpus normatif international structuré. La norme **ISA 315** (Identifying and Assessing the Risks of Material Misstatement) constitue la pierre angulaire de l'approche par les risques. Elle définit les procédures que l'auditeur doit mettre en œuvre pour identifier les risques d'anomalies significatives, qu'ils soient liés à la fraude ou à une erreur (IAASB, 2019).

Le risque d'audit se décompose en trois composantes :
1.  Le **risque inhérent**, lié à la nature de l'activité ;
2.  Le **risque lié au contrôle**, fonction de la robustesse du contrôle interne ;
3.  Le **risque de non-détection**, résiduel aux procédures de l'auditeur.

L'auditeur module l'étendue de ses travaux en fonction de l'évaluation de ces risques. En contexte marocain, l'OECCA publie des guides pratiques alignés sur les ISA. La formation du jugement professionnel reste largement tacite et dépendante de l'expérience sectorielle de l'auditeur (Filali Meknassi, 2018).

### II.2 Machine Learning appliqué à l'audit
L'application du Machine Learning à l'audit est un champ de recherche en croissance rapide depuis les années 2010. Les travaux pionniers de Koskivaara (2004) ont démontré la capacité des réseaux de neurones à détecter des anomalies dans les données comptables.

Ngai et al. (2011), dans une méta-analyse de 49 études, montrent que les techniques de Data Mining appliquées à la détection de fraudes financières atteignent des taux de précision supérieurs à 90% sur des jeux de données bien structurés. Leur revue identifie les arbres de décision, les SVM et les réseaux de neurones comme les algorithmes les plus performants.

Bao et al. (2020) proposent l'utilisation de RUSBoost pour la détection de fraudes dans les états financiers, avec des résultats surpassant significativement les modèles traditionnels (Beneish M-Score, Altman Z-Score).

### II.3 Interprétabilité des modèles
Un enjeu fondamental réside dans l'interprétabilité des modèles. Un commissaire aux comptes engage sa responsabilité civile et pénale : il ne peut pas se contenter d'une boîte noire. Lundberg & Lee (2017) ont introduit le framework **SHAP** (SHapley Additive exPlanations), qui permet d'attribuer à chaque variable une contribution marginale à la prédiction d'un modèle.

Rudin (2019), dans Nature Machine Intelligence, défend la supériorité des modèles intrinsèquement interprétables pour les domaines à fort enjeu. Ces travaux convergent vers une conclusion : le ML peut être un outil d'aide à la décision pertinent en audit, à condition que ses outputs soient intelligibles et soumis au jugement final du professionnel.

---

## III. Présentation du Dataset

### III.1 Source et mode de génération
Le dataset utilisé est un jeu de données simulé de 1 200 observations, construit à partir de règles métier inspirées des pratiques reconnues de l'évaluation du risque d'audit. Cette approche de simulation contrôlée est courante dans la recherche appliquée en audit lorsque des données réelles labellisées ne sont pas disponibles, en raison du secret professionnel.

La variable cible a été construite à partir d'un score de risque latent combinant les indicateurs quantitatifs, avec des ajustements par cycle et niveau de digitalisation, puis des seuils de classification définis. Un bruit gaussien a été ajouté pour reproduire la variabilité inhérente au jugement professionnel réel.

### III.2 Description des variables

| Variable | Type | Description | Référence ISA |
| :--- | :--- | :--- | :--- |
| **cycle** | Catégorielle | Cycle comptable audité (5 modalités) | ISA 315 — risques par processus |
| **volume_transactions** | Numérique | Nombre de transactions (proxy complexité) | ISA 315 §A27 |
| **nb_anomalies_passees** | Entier | Anomalies détectées en N-1 | ISA 315 §A28 |
| **controle_interne_score** | Numérique [0-1] | Score questionnaire CI normalisé | ISA 315 / ISA 265 |
| **complexite_si** | Entier [1-5] | Score de complexité du SI | ISA 315 §A65 |
| **digitalisation** | Numérique [0-1] | Niveau de digitalisation des processus | ISA 315 §A64 |
| **dso** | Numérique | Days Sales Outstanding (jours) | Ratio financier créances |
| **rotation_stocks** | Numérique | Ratio de rotation des stocks | ISA 501 |
| **marge_brute** | Numérique [0-1] | Marge brute normalisée | Indicateur de performance |
| **historique_audit** | Entier [0-4] | Score observations d'audit antérieures | ISA 300 — Planification |
| **risque** | Cible — 3 classes | Niveau de risque : faible/modéré/élevé | ISA 200 — Risque global |

### III.3 Analyse exploratoire (EDA)
Les graphiques ci-dessous présentent les principales caractéristiques du dataset.

*   **Figure 1 — Distribution de la variable cible (3 classes équilibrées — n=1 200)**
    Les trois classes sont parfaitement équilibrées (~33 % chacune), ce qui élimine le problème du déséquilibre des classes et permet l'utilisation directe de l'accuracy comme métrique principale.

*   **Figure 2 — Répartition du risque d'audit par cycle comptable**
    Le cycle Trésorerie présente une surreprésentation dans la classe « élevé », cohérente avec son statut de cycle le plus sensible aux fraudes par décaissement (ISA 240). Les Achats et les Ventes présentent également une proportion significative de risque élevé.

*   **Figure 3 — Score de contrôle interne par niveau de risque d'audit**
    La corrélation négative entre le score de contrôle interne et le niveau de risque est nette : les cycles classés « élevé » présentent systématiquement des scores CI plus faibles. Cette relation valide la logique métier sous-jacente au modèle.

*   **Figure 4 — Matrice de corrélation des variables numériques**
    La matrice de corrélation révèle des dépendances modérées entre variables. Le score de contrôle interne présente la corrélation la plus significative avec le niveau de risque. L'absence de multicolinéarité forte valide l'utilisation simultanée de toutes les variables.

---

## IV. Développement — Modélisation et Pipeline

### IV.1 Préparation des données et Feature Engineering
La préparation des données suit un pipeline structuré, implémenté via les classes `Pipeline` et `ColumnTransformer` de scikit-learn pour garantir l'absence de data leakage entre les ensembles d'entraînement et de test. Le partitionnement est 80/20 avec stratification.

Deux indicateurs composites à valeur métier ont été créés :
1.  `indice_vulnerabilite = volume_transactions / (score_controle_interne + 0.01)` — capture le rapport entre l'exposition volumétrique et la robustesse du contrôle interne.
2.  `alerte_historique = (nb_anomalies_passees > 2).astype(int)` — variable binaire signalant un historique d'anomalies préoccupant, analogue à un drapeau rouge en terminologie d'audit.

### IV.2 Algorithmes de classification comparés

| Algorithme | Famille | Justification |
| :--- | :--- | :--- |
| **Régression Logistique** | Modèle linéaire | Baseline interprétable — modèle de référence |
| **Random Forest** | Ensemble (Bagging) | Robustesse, gestion non-linéarité, importance native |
| **SVM** | Kernel methods | Performant sur données tabulaires de taille moyenne |
| **KNN** | Instance-based | Non paramétrique, sensible à la localité des données |

### IV.3 Optimisation et sélection du meilleur modèle
Chaque modèle est encapsulé dans un Pipeline scikit-learn (`preprocessor → classifier`). Une recherche par grille (`GridSearchCV`) avec validation croisée à 5 plis a été réalisée sur le Random Forest pour optimiser les hyperparamètres : `n_estimators ∈ {50, 100, 200}`, `max_depth ∈ {None, 10, 20}`, `min_samples_split ∈ {2, 5}`.

---

## V. Résultats et Discussion

### V.1 Performances comparatives des modèles
*   **Figure 5 — Comparaison des performances (Accuracy) par algorithme**

La Régression Logistique obtient la meilleure accuracy (63,8%) sur ce dataset réel, ce qui illustre un point important : sur des données réellement complexes, un modèle simple et interprétable peut surpasser des modèles plus sophistiqués. Le Random Forest (55,4%) et le KNN (53,3%) souffrent probablement d'overfitting avec les paramètres par défaut, tandis que le SVM (61,7%) offre un bon équilibre.

Ce résultat est précieux pour l'audit : il plaide en faveur de la Régression Logistique, modèle intrinsèquement interprétable, dont la logique peut être directement expliquée au commanditaire de la mission.

### V.2 Matrice de confusion — Interprétation
*   **Figure 6 — Matrice de confusion — Régression Logistique (jeu de test, n=240)**

L'analyse de la matrice de confusion est instructive sous l'angle de l'audit. Les confusions les plus fréquentes se situent entre les classes « élevé » et « modéré », ce qui constitue une erreur acceptable d'un point de vue professionnel : un cycle sous-évalué de « élevé » à « modéré » recevra tout de même des tests substantifs.

En revanche, les confusions « Faible ↔ Élevé » — les plus dangereuses en audit — sont rares, ce qui constitue une propriété essentielle pour un usage professionnel. Un faux négatif (risque élevé classé faible) conduirait à des diligences insuffisantes et exposerait l'auditeur à des risques de responsabilité.

### V.3 Importance des variables
*   **Figure 7 — Importance des variables (Random Forest — Gini Importance)**

L'analyse de l'importance des variables révèle la hiérarchie suivante :
1.  Le **score de contrôle interne** domine — confirmant le postulat fondamental de l'audit ISA 315 ;
2.  Le **nombre d'anomalies passées** constitue un signal d'alerte fort, en cohérence avec ISA 300 ;
3.  Les **indicateurs financiers** (DSO, rotation_stocks) contribuent significativement ;
4.  La variable engineerée `indice_vulnerabilite` valide sa pertinence métier.

### V.4 Interprétation métier des résultats
Les résultats sont en cohérence avec la doctrine d'audit internationale, ce qui valide la qualité des données simulées et la pertinence du modèle :

1.  **Score de contrôle interne prépondérant** — La norme ISA 315 identifie l'environnement de contrôle interne comme le fondement de l'évaluation du risque. Sa domination dans l'importance des variables constitue une validation interne du modèle.
2.  **Historique d'anomalies pénalisant** — La norme ISA 300 exige que l'auditeur prenne en compte les conclusions des missions précédentes. Le modèle reproduit automatiquement cette logique de continuité.
3.  **Sensibilité structurelle du cycle Trésorerie** — Universellement reconnu comme le cycle le plus exposé aux fraudes par décaissement, il ressort systématiquement à risque plus élevé dans le modèle.
4.  **Effet protecteur de la digitalisation** — Les SI digitalisés offrent meilleure traçabilité et contrôles automatisés, réduisant l'exposition aux erreurs humaines.

---

## VI. Limites et Axes d'Amélioration

L'analyse critique du projet conduit à identifier plusieurs limites importantes :

1.  **Données simulées** — Le dataset est construit sur des règles métier simplifiées. Des phénomènes complexes comme la collusion ou les fraudes comptables sophistiquées ne peuvent être modélisés sans données réelles.
2.  **Subjectivité résiduelle** — Le score de contrôle interne reste dépendant du questionnaire rempli par l'auditeur lors de la phase intérimaire, introduisant un biais difficilement réductible.
3.  **Absence de dimension temporelle** — Le modèle traite chaque observation indépendamment. Une architecture de type série temporelle permettrait de mieux capturer la dynamique du risque.
4.  **Performances modérées** — Sur le dataset réel, l'accuracy maximale atteinte est de 63,8%. Un tuning approfondi et l'intégration de données supplémentaires permettraient d'améliorer ce résultat.

**Axes d'amélioration futurs :**
1.  **Détection d'anomalies non supervisée (Isolation Forest)** — classifier les transactions individuelles du FEC pour une granularité plus fine.
2.  **NLP sur documents de gouvernance** — analyser les PV de Conseil d'Administration avec CamemBERT pour extraire le « ton de la direction ».
3.  **Déploiement en API REST** — packager le modèle avec MLflow ou FastAPI pour intégration dans les outils de gestion de mission.
4.  **Calibration probabiliste** — utiliser `CalibratedClassifierCV` pour des probabilités fiables et exploitables.

---

## VII. Conclusion

Ce projet a démontré la faisabilité et la pertinence d'une approche de Data Science appliquée à la classification des risques d'audit par cycle comptable. En comparant quatre algorithmes de Machine Learning supervisé sur un dataset de 1 200 observations, il a été établi que la **Régression Logistique** offre les meilleures performances sur ce dataset structuré réel, avec une accuracy de 63,8%.

Au-delà de la performance brute, l'analyse de l'importance des variables a permis de produire un modèle dont la logique interne est cohérente avec les principes fondamentaux des normes ISA. Le score de contrôle interne, l'historique des anomalies et le cycle de Trésorerie ressortent systématiquement comme les facteurs de risque les plus déterminants, en parfaite adéquation avec la doctrine d'audit internationale.

Ce travail illustre un cas concret de convergence entre la filière CAC et les compétences numériques. L'auditeur de demain ne sera pas remplacé par un algorithme, mais celui qui maîtrise ces outils disposera d'un avantage compétitif décisif pour planifier ses missions efficacement et objectiver ses évaluations de risque. Les perspectives d'amélioration — notamment l'analyse granulaire des écritures comptables et l'intégration du NLP — ouvrent des voies de recherche stimulantes à la frontière entre l'intelligence artificielle et la profession comptable réglementée.

---

## Bibliographie

**Normes et textes réglementaires**
1.  IAASB. (2019). *ISA 200 — Objectifs généraux de l'auditeur indépendant*. IFAC.
2.  IAASB. (2019). *ISA 315 (Révisée) — Identification et évaluation des risques d'anomalies significatives*. IFAC.
3.  IAASB. (2009). *ISA 240 — La responsabilité de l'auditeur dans la prise en compte de fraudes*. IFAC.
4.  IAASB. (2009). *ISA 265 — Communication des faiblesses du contrôle interne*. IFAC.
5.  IAASB. (2009). *ISA 300 — Planification d'un audit d'états financiers*. IFAC.
6.  OECCA Maroc. (2020). *Guide pratique d'application des normes ISA au contexte marocain*.

**Articles scientifiques**
1.  Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting accounting fraud in publicly traded US firms using a machine learning approach. *Journal of Accounting Research*, 58(1), 199–235.
2.  Koskivaara, E. (2004). Artificial neural networks in analytical review procedures. *Managerial Auditing Journal*, 19(2), 191–223.
3.  Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30, 4765–4774.
4.  Ngai, E. W. T., et al. (2011). The application of data mining techniques in financial fraud detection. *Decision Support Systems*, 50(3), 559–569.
5.  Rudin, C. (2019). Stop explaining black box ML models for high stakes decisions. *Nature Machine Intelligence*, 1(5), 206–215.

**Ouvrages**
1.  Filali Meknassi, R. (2018). *L'audit financier au Maroc : pratiques, normes et perspectives*. Éditions Toubkal, Casablanca.
2.  Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3e éd.). O'Reilly Media.
3.  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785–794.

---

## Annexes

### Annexe A — Extrait du dataset (5 premières lignes)

| cycle | volume_trx | nb_anomalies | ctrl_interne | complexite | dso | rotation | marge | hist_audit | risque |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| tresorerie | 3 729 | 2 | 0,761 | 2 | 41,63 | 6,13 | 0,415 | 0 | faible |
| stocks | 4 325 | 3 | 0,617 | 1 | 47,73 | 6,92 | 0,345 | 0 | faible |
| paie | 3 167 | 5 | 0,786 | 4 | 17,69 | 0,90 | 0,440 | 0 | eleve |

### Annexe B — Architecture du pipeline scikit-learn

```text
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num: SimpleImputer(median) → StandardScaler()
│   └── cat: SimpleImputer(constant) → OneHotEncoder()
└── classifier
    └── LogisticRegression / RandomForest / SVC / KNN
```

### Annexe C — Rapport de classification détaillé (Régression Logistique)

| Classe | Précision | Rappel | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Élevé** | 0,70 | 0,68 | 0,69 | 82 |
| **Faible** | 0,70 | 0,75 | 0,72 | 79 |
| **Modéré** | 0,50 | 0,48 | 0,49 | 79 |
| **Macro avg** | 0,63 | 0,64 | 0,64 | 240 |

*Note : La classe « Modéré » est la plus difficile à prédire, ce qui est cohérent : en audit, les cycles à risque modéré présentent des profils mixtes qui chevauchent les caractéristiques des deux autres classes.*

---

**Kaiss · Salma · Hamdoune · Oumaima | CAC L3 S8 | 2024–2025**
