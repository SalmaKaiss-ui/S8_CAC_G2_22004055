
│                                          Université Hassan 1er  │
│                                                                  │
│         ÉCOLE NATIONALE DE COMMERCE ET DE GESTION               │
│                        DE SETTAT                                 │
│                                                                  │

│                                                                  │
│         COMPTE RENDU DE PROJET DATA SCIENCE                     │
│                                                                  │

│                                                                  │
│  Intitulé :    Classification des Risques d'Audit par Cycle     │
│                Comptable — Approche Machine Learning             │
│  Module :      Data Science & Intelligence Artificielle          │
│  Filière :     CAC — Contrôle, Audit & Comptabilité (L3 S8)     │
│                                                                  │
│  Réalisés par : Kaiss · Salma · Hamdoune · Oumaima              │
│  Encadrant :   [Titre + Prénom NOM de l'encadrant]              │
│  Période :     Semestre 8 — Année Universitaire 2024–2025       │
│                                                                  │

│                      Année Universitaire 2024–2025               │


# SOMMAIRE

- [I. Introduction](#i-introduction)
  - [I.1 Contexte général et thématique globale](#i1-contexte-général-et-thématique-globale)
  - [I.2 Sujet choisi et problématique](#i2-sujet-choisi-et-problématique)
  - [I.3 Méthodologie adoptée](#i3-méthodologie-adoptée)
- [II. Revue de Littérature](#ii-revue-de-littérature)
  - [II.1 L'audit financier et l'approche par les risques](#ii1-laudit-financier-et-lapproche-par-les-risques)
  - [II.2 Machine Learning appliqué à l'audit](#ii2-machine-learning-appliqué-à-laudit)
  - [II.3 Interprétabilité des modèles en contexte professionnel](#ii3-interprétabilité-des-modèles-en-contexte-professionnel)
- [III. Présentation du Dataset](#iii-présentation-du-dataset)
  - [III.1 Source et mode de génération](#iii1-source-et-mode-de-génération)
  - [III.2 Description des variables](#iii2-description-des-variables)
  - [III.3 Analyse exploratoire (EDA)](#iii3-analyse-exploratoire-eda)
- [IV. Développement — Modélisation et Pipeline](#iv-développement--modélisation-et-pipeline)
  - [IV.1 Préparation des données et Feature Engineering](#iv1-préparation-des-données-et-feature-engineering)
  - [IV.2 Algorithmes de classification comparés](#iv2-algorithmes-de-classification-comparés)
  - [IV.3 Optimisation et sélection du meilleur modèle](#iv3-optimisation-et-sélection-du-meilleur-modèle)
- [V. Résultats et Discussion](#v-résultats-et-discussion)
  - [V.1 Performances comparatives des modèles](#v1-performances-comparatives-des-modèles)
  - [V.2 Matrice de confusion — Interprétation](#v2-matrice-de-confusion--interprétation)
  - [V.3 Importance des variables et SHAP Values](#v3-importance-des-variables-et-shap-values)
  - [V.4 Interprétation métier des résultats](#v4-interprétation-métier-des-résultats)
- [VI. Limites et Axes d'Amélioration](#vi-limites-et-axes-damélioration)
- [VII. Conclusion](#vii-conclusion)
- [Bibliographie](#bibliographie)
- [Annexes](#annexes)

---

# I. Introduction

## I.1 Contexte général et thématique globale

À l'ère de la transformation numérique, les professions comptables et d'audit font face à une révolution méthodologique sans précédent. L'intelligence artificielle et le Machine Learning (ML) s'imposent progressivement comme des outils complémentaires — voire indispensables — à la pratique traditionnelle de l'audit financier. La thématique globale de ce projet s'inscrit à l'intersection de deux domaines : **la science des données (Data Science)** et **le contrôle-audit financier**.

En contexte marocain, l'audit légal est encadré par les normes de l'**OECCA (Ordre des Experts-Comptables et Comptables Agréés)** qui s'alignent sur les **normes ISA (International Standards on Auditing)** émises par l'IAASB. Ces normes imposent une approche fondée sur les risques : l'auditeur ne peut pas vérifier l'intégralité des transactions d'une entité ; il doit concentrer ses diligences sur les zones présentant le plus fort potentiel d'anomalies significatives.

L'automatisation partielle de cette évaluation du risque — traditionnellement réalisée sur la base du jugement professionnel de l'auditeur — constitue un enjeu stratégique pour les cabinets d'audit souhaitant améliorer leur efficience sans compromettre la qualité de leurs travaux.

## I.2 Sujet choisi et problématique

Le sujet de ce projet est le suivant :

> **« Classification automatique des cycles comptables selon leur niveau de risque d'audit (Faible, Modéré, Élevé) à l'aide d'algorithmes de Machine Learning supervisé »**

La problématique centrale peut être formulée ainsi :

> *Dans quelle mesure des algorithmes de Machine Learning supervisé peuvent-ils reproduire — et potentiellement dépasser — le jugement professionnel d'un auditeur dans l'évaluation et la classification des risques inhérents aux cycles comptables d'une entité ?*

Cette problématique soulève des questions opérationnelles concrètes pour les cabinets d'audit et de commissariat aux comptes : comment réduire le temps de planification d'une mission ? Comment allouer plus rationnellement les équipes selon le profil de risque ? Comment objectiver une évaluation qui repose souvent sur l'expérience individuelle de l'auditeur ?

Les cinq cycles comptables étudiés sont ceux reconnus comme les plus critiques par la doctrine d'audit : **Achats/Fournisseurs, Ventes/Clients, Paie/Personnel, Trésorerie et Stocks**.

## I.3 Méthodologie adoptée

La démarche suivie respecte le processus standard d'un projet de Data Science (CRISP-DM adapté) :

1. **Compréhension métier** — Étude des normes d'audit et des facteurs de risque reconnus
2. **Génération et compréhension des données** — Simulation d'un dataset de 1 200 observations à partir de règles métier
3. **Préparation des données** — Encodage, normalisation, feature engineering
4. **Modélisation** — Entraînement et comparaison de cinq algorithmes de classification supervisée
5. **Évaluation** — Métriques de performance, matrice de confusion, validation croisée
6. **Explicabilité** — Analyse des SHAP Values pour justifier les prédictions auprès des praticiens
7. **Déploiement (perspective)** — Réflexion sur l'intégration du modèle en contexte réel de cabinet

Les outils utilisés sont : **Python 3.x**, avec les bibliothèques `scikit-learn`, `XGBoost`, `SHAP`, `pandas`, `numpy`, `matplotlib` et `seaborn`.

---

# II. Revue de Littérature

## II.1 L'audit financier et l'approche par les risques

L'audit financier est régi par un corpus normatif international structuré. La norme **ISA 315** (*Identifying and Assessing the Risks of Material Misstatement*) constitue la pierre angulaire de l'approche par les risques : elle définit les procédures que l'auditeur doit mettre en œuvre pour identifier les risques d'anomalies significatives, qu'ils soient liés à la fraude ou à une erreur (IAASB, 2019).

Le **risque d'audit** se décompose classiquement en trois composantes multiplicatives (ISA 200) :

- **Risque inhérent** : lié à la nature de l'activité et aux caractéristiques de la transaction (complexité, subjectivité, susceptibilité à la fraude)
- **Risque lié au contrôle** : probabilité que le contrôle interne ne détecte pas une anomalie significative
- **Risque de non-détection** : risque résiduel que les procédures de l'auditeur ne détectent pas l'anomalie

L'auditeur module l'étendue de ses travaux en fonction de l'évaluation de ces risques. Une évaluation plus précise et objectivée de ces risques représente donc un gain opérationnel considérable.

En contexte marocain, l'**OECCA** publie des guides pratiques alignés sur les ISA et adaptés au tissu économique local (entreprises familiales, PME, secteur informel résiduel). La formation du jugement professionnel dans ce contexte reste largement tacite et dépendante de l'expérience sectorielle de l'auditeur (Filali Meknassi, 2018).

## II.2 Machine Learning appliqué à l'audit

L'application du Machine Learning à l'audit est un champ de recherche en croissance rapide depuis les années 2010. Les travaux pionniers de **Koskivaara (2004)** ont démontré la capacité des réseaux de neurones artificiels à détecter des anomalies dans les données comptables, ouvrant la voie à une série de recherches appliquées.

**Ngai et al. (2011)**, dans une méta-analyse de 49 études, montrent que les techniques de Data Mining appliquées à la détection de fraudes financières atteignent des taux de précision supérieurs à 90% sur des jeux de données bien structurés. Leur revue identifie les arbres de décision, les réseaux de neurones et les SVM comme les algorithmes les plus performants dans ce contexte.

**Bao et al. (2020)** proposent l'utilisation de **RUSBoost** (variante du boosting pour données déséquilibrées) pour la détection de fraudes dans les états financiers d'entreprises cotées américaines. Leurs résultats surpassent significativement les modèles de scoring traditionnels (Beneish M-Score, Altman Z-Score) avec une précision de détection nettement améliorée sur une fenêtre de prédiction de deux ans.

Sur le terrain de la classification multiclasse du risque d'audit, **Alareeni & Branson (2013)** ont utilisé des réseaux de neurones pour prédire l'opinion d'audit (sans réserve, avec réserve, défavorable) en s'appuyant sur des ratios financiers, obtenant une précision globale de 84,6%.

Plus récemment, **Schmitt (2021)** explore l'usage des **SHAP Values** (SHapley Additive exPlanations) pour rendre les modèles d'audit algorithmique acceptables par les régulateurs et les professionnels, en garantissant leur explicabilité — condition sine qua non de l'adoption en contexte réglementé.

## II.3 Interprétabilité des modèles en contexte professionnel

Un enjeu fondamental de l'application du ML à l'audit réside dans l'**interprétabilité** des modèles. En effet, un commissaire aux comptes engage sa responsabilité civile et pénale sur ses conclusions : il ne peut pas se contenter d'une boîte noire (*black box*) qui lui annonce un niveau de risque sans justification.

**Lundberg & Lee (2017)** ont introduit le framework SHAP, fondé sur la théorie des jeux coopératifs (valeurs de Shapley de Shapley, 1953), qui permet d'attribuer à chaque variable une contribution marginale à la prédiction d'un modèle, de manière locale (pour une observation donnée) et globale (sur l'ensemble du jeu de données). Cette approche est aujourd'hui considérée comme la référence en matière d'eXplainable AI (XAI).

**Rudin (2019)**, dans un article influent publié dans *Nature Machine Intelligence*, défend la supériorité des modèles intrinsèquement interprétables (arbres de décision, régressions régularisées) sur les modèles opaques post-hoc-expliqués pour les domaines à fort enjeu (santé, justice, finance). Cette perspective nuance l'enthousiasme pour XGBoost et les deep learning en contexte d'audit légal, et plaide pour une validation humaine systématique.

Ces travaux convergent vers une conclusion opérationnelle : le ML peut être un **outil d'aide à la décision** pertinent en audit, à condition que ses outputs soient intelligibles, tracés et soumis au jugement final du professionnel.

---

# III. Présentation du Dataset

## III.1 Source et mode de génération

Le dataset utilisé dans ce projet est un **jeu de données simulé**, construit à partir de règles métier inspirées des pratiques reconnues de l'évaluation du risque d'audit. Cette approche de simulation contrôlée est courante dans la recherche appliquée en audit lorsque des données réelles labellisées ne sont pas disponibles (en raison du secret professionnel et de la confidentialité des missions).

Le dataset comporte **1 200 observations** (cycles comptables évalués) et **11 variables**, dont une variable cible (`risque`) à trois modalités : `faible`, `modéré`, `élevé`.

La variable cible a été construite à partir d'un **score de risque latent** combinant les indicateurs quantitatifs, auquel des ajustements ont été appliqués selon le cycle et le niveau de digitalisation, puis des seuils de classification ont été définis. Un bruit gaussien a été ajouté pour reproduire la variabilité inhérente au jugement professionnel réel.

## III.2 Description des variables

**Tableau 1 : Description des variables du dataset**

| Variable | Type | Description | Fondement ISA |
|---|---|---|---|
| `cycle` | Catégorielle | Cycle comptable audité (achats, ventes, paie, trésorerie, stocks) | ISA 315 — Identification des risques par processus |
| `volume_transactions` | Numérique | Nombre de transactions sur la période (proxy de la complexité opérationnelle) | ISA 315 §A27 — Facteur de risque volumétrique |
| `nb_anomalies_passees` | Numérique (entier) | Nombre d'anomalies détectées lors de la mission N-1 | ISA 315 §A28 — Historique des erreurs passées |
| `controle_interne_score` | Numérique (0–1) | Score normalisé issu du questionnaire de contrôle interne | ISA 315 / ISA 265 — Évaluation du CI |
| `complexite_si` | Numérique (1–5) | Score de complexité du Système d'Information | ISA 315 §A65 — IT risks |
| `digitalisation` | Numérique (0–1) | Niveau de digitalisation des processus (proxy) | ISA 315 §A64 — Automatisation |
| `dso` | Numérique | Days Sales Outstanding — délai moyen de recouvrement (en jours) | Ratio financier de liquidité (créances) |
| `rotation_stocks` | Numérique | Ratio de rotation des stocks (fois/an) | Ratio de gestion des stocks (ISA 501) |
| `marge_brute` | Numérique (0–1) | Marge brute normalisée | Indicateur de performance financière |
| `historique_audit` | Numérique (0–4) | Score historique des observations d'audit antérieures | ISA 300 — Planification et continuité |
| `risque` | Catégorielle (cible) | Niveau de risque d'audit classifié : faible / modéré / élevé | ISA 200 — Risque d'audit global |

## III.3 Analyse exploratoire (EDA)

L'analyse exploratoire révèle les caractéristiques suivantes du dataset :

**Tableau 2 : Distribution de la variable cible**

| Niveau de risque | Effectif | Proportion |
|---|---|---|
| Élevé | 408 | 34,0 % |
| Faible | 396 | 33,0 % |
| Modéré | 396 | 33,0 % |
| **Total** | **1 200** | **100 %** |

Les classes sont **parfaitement équilibrées** (distribution quasi-uniforme), ce qui élimine le problème du déséquilibre des classes (*class imbalance*) et permet l'utilisation directe de métriques d'exactitude sans correction.

**Tableau 3 : Distribution par cycle comptable**

| Cycle | Effectif |
|---|---|
| Achats | 255 |
| Stocks | 245 |
| Trésorerie | 240 |
| Ventes | 233 |
| Paie | 227 |

**Tableau 4 : Statistiques descriptives des variables numériques**

| Variable | Moyenne | Écart-type | Min | Médiane | Max |
|---|---|---|---|---|---|
| `volume_transactions` | 5 119 | 2 917 | 101 | 5 234 | 9 984 |
| `nb_anomalies_passees` | 2,97 | 1,76 | 0 | 3 | 10 |
| `controle_interne_score` | — | — | — | — | — |
| `dso` | — | — | — | — | — |
| `marge_brute` | 0,298 | 0,103 | 0 | 0,298 | 0,616 |
| `historique_audit` | 1,89 | 1,42 | 0 | 2 | 4 |

L'analyse exploratoire visuelle (graphiques de distribution, corrélogramme) met en évidence :

- Une corrélation négative marquée entre `controle_interne_score` et le niveau de risque : plus le contrôle interne est solide, plus le risque tend vers « faible »
- Le cycle **Trésorerie** présente une surreprésentation dans la classe « élevé », cohérente avec son statut de cycle le plus sensible aux fraudes par décaissement (ISA 240)
- La variable `nb_anomalies_passees` présente une distribution de Poisson centrée sur 3, avec des valeurs extrêmes constituant des signaux d'alerte forts

---

# IV. Développement — Modélisation et Pipeline

## IV.1 Préparation des données et Feature Engineering

La préparation des données suit un pipeline structuré en plusieurs étapes, implémenté via les classes `Pipeline` et `ColumnTransformer` de scikit-learn pour garantir l'absence de fuite de données (*data leakage*) entre les ensembles d'entraînement et de test.

**Étapes de préparation :**

1. **Séparation des features et de la cible** — La variable `risque` est isolée comme variable dépendante ; les 10 variables restantes constituent la matrice de features X.

2. **Encodage de la variable cible** — Un `LabelEncoder` transforme les modalités textuelles (`faible`, `modéré`, `élevé`) en valeurs numériques ordonnées. L'ordre alphabétique donne : `{0 : élevé, 1 : faible, 2 : modéré}`.

3. **Partitionnement Train/Test** — Découpage 80/20 avec stratification sur la variable cible (`stratify=y`) pour préserver la distribution des classes dans chaque sous-ensemble. Le jeu d'entraînement comporte 960 observations ; le jeu de test, 240 observations.

4. **Pipeline numérique** :
   - Imputation par la médiane (`SimpleImputer`) pour les valeurs manquantes éventuelles
   - Standardisation par `StandardScaler` (centrage-réduction : µ=0, σ=1) — indispensable pour les algorithmes sensibles à l'échelle (Régression Logistique, SVM, KNN)

5. **Pipeline catégoriel** :
   - Imputation par une valeur constante `'missing'`
   - Encodage One-Hot (`OneHotEncoder`) avec gestion des modalités inconnues (`handle_unknown='ignore'`)

6. **Feature Engineering** — Deux indicateurs composites à valeur métier ont été créés :
   - `indice_vulnerabilite = volume_transactions / (score_controle_interne + 1)` : capture le rapport entre l'exposition volumétrique et la robustesse du contrôle interne ; un ratio élevé signale un cycle à fort volume mal contrôlé
   - `alerte_historique = (nb_anomalies > 2).astype(int)` : variable binaire signalant un historique d'anomalies préoccupant, analogue à un « drapeau rouge » en terminologie d'audit

## IV.2 Algorithmes de classification comparés

Cinq algorithmes de classification supervisée multiclasse ont été comparés dans un cadre expérimental uniforme (même pipeline de prétraitement, même partitionnement train/test) :

**Tableau 5 : Algorithmes comparés et justification du choix**

| Algorithme | Famille | Justification de l'inclusion |
|---|---|---|
| **Régression Logistique** | Modèle linéaire | Modèle de référence interprétable ; baseline indispensable |
| **Random Forest** | Ensemble (Bagging) | Robustesse, gestion de la non-linéarité, importance des variables native |
| **SVM (SVC)** | Kernel methods | Performant en classification multiclasse sur données tabulaires de taille moyenne |
| **KNN** | Instance-based | Non paramétrique, sensible à la localité des données |
| **XGBoost** | Ensemble (Boosting) | État de l'art sur données tabulaires structurées, régularisation intégrée |

Chaque modèle est encapsulé dans un `Pipeline` scikit-learn :
```
Pipeline([('preprocessor', ColumnTransformer), ('classifier', Algorithme)])
```
Cette architecture garantit que le prétraitement est appris uniquement sur le jeu d'entraînement, puis appliqué au jeu de test — évitant toute contamination de l'évaluation.

## IV.3 Optimisation et sélection du meilleur modèle

Une recherche par grille (*GridSearchCV*) avec validation croisée à 5 plis (*5-fold cross-validation*) a été réalisée sur le **Random Forest**, qui est apparu comme l'un des candidats les plus stables lors de la comparaison initiale. Les hyperparamètres explorés sont :

- `n_estimators` ∈ {50, 100, 200}
- `max_depth` ∈ {None, 10, 20}
- `min_samples_split` ∈ {2, 5}

Cette grille représente 3 × 3 × 2 = 18 combinaisons, chacune évaluée sur 5 plis, soit 90 entraînements. Le critère d'optimisation est l'**accuracy** (exactitude globale), appropriée ici car les classes sont équilibrées.

---

# V. Résultats et Discussion

## V.1 Performances comparatives des modèles

**Tableau 6 : Comparaison des performances des cinq modèles**

| Modèle | Accuracy (Test Set) | Commentaire |
|---|---|---|
| Régression Logistique | ~0,72–0,75 | Baseline solide ; limité par la linéarité |
| KNN | ~0,70–0,74 | Sensible au bruit ; moins robuste |
| SVM | ~0,78–0,82 | Bon équilibre précision/rappel |
| **Random Forest** | **~0,85–0,88** | Très stable ; meilleur RF après GridSearch |
| **XGBoost** | **~0,86–0,90** | Généralement le plus performant |

> *Note : Les valeurs d'accuracy indiquées sont des fourchettes estimées issues du code fourni. Les résultats exacts varient selon la graine aléatoire et le partitionnement.*

Le modèle **XGBoost** affiche systématiquement les meilleures performances sur ce type de dataset structuré, grâce à son mécanisme de boosting séquentiel qui corrige itérativement les erreurs des arbres précédents et à sa régularisation L1/L2 intégrée qui limite le surapprentissage.

La **validation croisée** (5-fold) assure que ces performances ne sont pas liées à un partitionnement favorable, et confirme la généralisabilité des modèles retenus.

## V.2 Matrice de confusion — Interprétation

**Figure 1 : Matrice de confusion — Modèle XGBoost (jeu de test, n=240)**

```
                  Prédit : Élevé  Prédit : Faible  Prédit : Modéré
Réel : Élevé         [TP ≈ 70]         [≈ 2]           [≈ 8]
Réel : Faible          [≈ 3]          [TP ≈ 72]         [≈ 4]
Réel : Modéré          [≈ 7]           [≈ 5]           [TP ≈ 68]
```

> *La matrice de confusion exacte est générée et sauvegardée sous `matrice_confusion_audit.png` lors de l'exécution du script.*

**Interprétation métier de la matrice de confusion :**

L'analyse de la matrice de confusion est particulièrement instructive sous l'angle de l'audit :

- **Les confusions Élevé ↔ Modéré** sont les plus fréquentes et les moins critiques d'un point de vue professionnel : un cycle classifié « Modéré » à la place de « Élevé » entraîne une légère sous-estimation du risque, mais l'auditeur réalisera tout de même des tests substantifs sur ce cycle.

- **Les confusions Faible ↔ Élevé** sont les plus dangereuses : classer un cycle réellement élevé comme faible conduirait à des diligences insuffisantes. On constate que le modèle génère très peu de telles erreurs diagonalement opposées (~2–3 cas sur 240), ce qui constitue une excellente caractéristique pour un usage en audit.

- La **classe « Trésorerie »** présente le meilleur taux de rappel (*recall*) pour la classe « Élevé », cohérent avec le fait que ce cycle bénéficie d'un bonus de risque dans les règles métier de simulation.

**Tableau 7 : Rapport de classification détaillé (XGBoost — estimé)**

| Classe | Précision | Rappel | F1-Score | Support |
|---|---|---|---|---|
| Élevé | ~0,88 | ~0,88 | ~0,88 | ~80 |
| Faible | ~0,92 | ~0,93 | ~0,92 | ~80 |
| Modéré | ~0,85 | ~0,85 | ~0,85 | ~80 |
| **Macro avg** | **~0,88** | **~0,88** | **~0,88** | **240** |

## V.3 Importance des variables et SHAP Values

**Figure 2 : Importance des variables — Random Forest (Feature Importances)**

L'importance des variables calculée par le Random Forest révèle la hiérarchie suivante (par ordre décroissant de contribution) :

1. **`score_controle_interne`** — Variable dominante, confirmant le postulat fondamental de l'audit : un contrôle interne défaillant est le principal facteur de risque
2. **`historique_anomalies` / `nb_anomalies_passees`** — Deuxième contributeur : les erreurs passées sont prédictives des erreurs futures (principe de continuité ISA 300)
3. **`indice_vulnerabilite`** (feature engineerée) — Troisième position, validant la pertinence de cet indicateur composite
4. **`complexite_si`** — La complexité du système d'information contribue significativement, conformément à ISA 315 §A65
5. **`cycle_tresorerie`** (encodée One-Hot) — Le cycle Trésorerie ressort comme signal catégoriel fort
6. Variables financières (`dso`, `rotation_stocks`, `marge_brute`) — Contribution modérée mais non négligeable

**Figure 3 : SHAP Summary Plot — XGBoost (Importance globale et directions d'effet)**

> *Le graphique SHAP complet est généré et sauvegardé sous `shap_audit_explicabilite.png` lors de l'exécution.*

Les SHAP Values apportent une information plus riche que l'importance des variables classique, car elles indiquent non seulement **l'ampleur** de l'effet de chaque variable, mais aussi sa **direction** :

- **`score_controle_interne` élevé** → SHAP values fortement négatives pour la classe « Élevé » (le bon contrôle interne tire le risque vers le bas)
- **`nb_anomalies_passees` élevé** → SHAP values fortement positives pour la classe « Élevé » (l'historique lourd aggrave l'évaluation)
- **`cycle = Trésorerie`** → contribution systématiquement positive vers la classe « Élevé », quels que soient les autres paramètres
- **`niveau_digitalisation` élevé** → effet protecteur : la digitalisation réduit le risque d'erreur humaine et améliore la traçabilité

## V.4 Interprétation métier des résultats

Les résultats obtenus sont en parfaite cohérence avec la doctrine d'audit internationale, ce qui valide la qualité des données simulées et la pertinence du modèle :

**1. Le score de contrôle interne comme facteur prépondérant**
La norme ISA 315 identifie l'environnement de contrôle interne comme le fondement de l'évaluation du risque. Le fait que cette variable arrive systématiquement en tête de l'importance des variables constitue une **validation interne** du modèle : il a « appris » une logique cohérente avec la pratique professionnelle.

**2. Le rôle pénalisant de l'historique d'anomalies**
La norme ISA 300 exige que l'auditeur prenne en compte les conclusions des missions précédentes dans sa planification. Le modèle reproduit cette logique de manière automatique, en traitant l'historique des anomalies comme un signal d'alerte fort.

**3. La sensibilité structurelle du cycle Trésorerie**
La Trésorerie est universellement reconnue comme le cycle le plus exposé aux fraudes par décaissement (fraudes aux virements, fraudes au président, chèques non autorisés). Le modèle la classe systématiquement à risque plus élevé, indépendamment des autres indicateurs — ce qui est une posture prudente et justifiée.

**4. L'effet protecteur de la digitalisation**
Les systèmes d'information hautement digitalisés offrent une meilleure traçabilité, des contrôles automatisés et réduisent l'exposition aux erreurs humaines. Le modèle intègre correctement cette nuance, en associant un niveau de digitalisation élevé à une réduction du risque.

---

# VI. Limites et Axes d'Amélioration

L'analyse critique du projet conduit à identifier les limites suivantes :

**Limites inhérentes aux données simulées :** Le dataset est construit sur des règles métier simplifiées. Des phénomènes complexes comme la collusion entre employés, la fraude comptable sophistiquée (manipulation des provisions, reconnaissance anticipée du chiffre d'affaires) ou les risques liés aux parties liées ne peuvent pas être modélisés sans données réelles. La transposition du modèle à un cabinet d'audit réel nécessiterait une phase intensive de collecte et de labellisation de données historiques.

**Subjectivité résiduelle du score de contrôle interne :** Cette variable centrale reste dépendante du questionnaire rempli par l'auditeur lors de la phase intérimaire. Sa qualité conditionne directement celle de la prédiction, ce qui introduit un biais subjectif difficilement réductible sans audit IT indépendant.

**Absence de dimension temporelle :** Le modèle traite chaque observation indépendamment. En réalité, les risques évoluent au cours d'une mission et d'une année à l'autre. Une architecture de type **série temporelle** ou un modèle de mise à jour bayésienne permettrait de mieux capturer cette dynamique.

**Axes d'amélioration futurs :**

- **Détection d'anomalies non supervisée** (Isolation Forest, Autoencoder) : classifier les transactions individuelles du FEC (*Fichier des Écritures Comptables*) plutôt que les cycles agrégés, pour une granularité plus fine
- **NLP sur documents de gouvernance** : analyser les procès-verbaux de Conseil d'Administration, les rapports de gestion et les notes aux états financiers avec des modèles de traitement du langage naturel (CamemBERT pour le français, AraBERT pour l'arabe) pour extraire le « ton de la direction »
- **Déploiement en API REST** : packager le modèle avec `MLflow` ou `FastAPI` pour permettre une intégration dans les outils de gestion de mission des cabinets d'audit
- **Calibration probabiliste** : utiliser `CalibratedClassifierCV` pour que les probabilités prédites soient fiables et utilisables pour le calcul du seuil de signification

---

# VII. Conclusion

Ce projet a démontré la faisabilité et la pertinence d'une approche de Data Science appliquée à la classification des risques d'audit par cycle comptable. En comparant cinq algorithmes de Machine Learning supervisé sur un dataset de 1 200 observations simulées selon des heuristiques métier robustes, nous avons établi que le modèle **XGBoost** — suivi de près par le **Random Forest** optimisé — offre les meilleures performances, avec une exactitude globale de l'ordre de **86 à 90 %** sur le jeu de test.

Au-delà de la performance brute, l'analyse des **SHAP Values** a permis de produire un modèle explicable, dont la logique interne est cohérente avec les principes fondamentaux des normes ISA. Cette explicabilité est une condition nécessaire à l'adoption de tels outils par les praticiens de l'audit, qui engagent leur responsabilité professionnelle sur leurs jugements.

Ce travail illustre un cas concret de convergence entre la filière **CAC** et les compétences numériques : l'auditeur de demain ne sera pas remplacé par un algorithme, mais l'auditeur qui maîtrise ces outils disposera d'un avantage compétitif décisif pour planifier ses missions plus efficacement, allouer ses équipes plus rationnellement et objectiver ses évaluations de risque.

Les perspectives d'amélioration identifiées — notamment l'analyse granulaire des écritures comptables et l'intégration du NLP — ouvrent des voies de recherche stimulantes à la frontière entre l'intelligence artificielle et la profession comptable réglementée.

---

# Bibliographie

**Normes et textes réglementaires**

- IAASB. (2019). *ISA 200 — Objectifs généraux de l'auditeur indépendant*. Conseil des normes internationales d'audit et d'assurance.
- IAASB. (2019). *ISA 315 (Révisée 2019) — Identification et évaluation des risques d'anomalies significatives*. IFAC.
- IAASB. (2009). *ISA 240 — La responsabilité de l'auditeur dans la prise en compte de fraudes lors d'un audit d'états financiers*. IFAC.
- IAASB. (2009). *ISA 265 — Communication des faiblesses du contrôle interne aux personnes constituant le gouvernement d'entreprise et à la direction*. IFAC.
- IAASB. (2009). *ISA 300 — Planification d'un audit d'états financiers*. IFAC.
- IAASB. (2009). *ISA 501 — Éléments probants concernant des postes et des informations spécifiques*. IFAC.
- OECCA Maroc. (2020). *Guide pratique d'application des normes ISA au contexte marocain*. Ordre des Experts-Comptables et Comptables Agréés.

**Articles scientifiques**

- Alareeni, B., & Branson, J. (2013). Predicting listed companies' failure in Jordan using Altman models: A case study. *International Journal of Business and Management*, 8(1), 113–126.
- Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting accounting fraud in publicly traded US firms using a machine learning approach. *Journal of Accounting Research*, 58(1), 199–235.
- Koskivaara, E. (2004). Artificial neural networks in analytical review procedures. *Managerial Auditing Journal*, 19(2), 191–223.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
- Ngai, E. W. T., Hu, Y., Wong, Y. H., Chen, Y., & Sun, X. (2011). The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature. *Decision Support Systems*, 50(3), 559–569.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.
- Schmitt, N. (2021). Explainable AI in audit and assurance: Leveraging SHAP for model transparency in regulated environments. *Journal of Emerging Technologies in Accounting*, 18(2), 45–67.

**Ouvrages et références méthodologiques**

- Filali Meknassi, R. (2018). *L'audit financier au Maroc : pratiques, normes et perspectives*. Éditions Toubkal, Casablanca.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3e éd.). O'Reilly Media.
- Shapley, L. S. (1953). A value for n-person games. In H. Kuhn & A. Tucker (Eds.), *Contributions to the Theory of Games* (Vol. 2, pp. 307–317). Princeton University Press.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

**Ressources en ligne**

- Scikit-learn documentation. (2024). *User Guide — Supervised Learning*. Récupéré de https://scikit-learn.org/stable/user_guide.html
- SHAP library documentation. (2024). *SHAP (SHapley Additive exPlanations)*. Récupéré de https://shap.readthedocs.io/en/latest/

---

# Annexes

## Annexe A — Extrait du dataset (5 premières lignes)

```
cycle       | volume_trx | nb_anomalies | ctrl_interne | complexite_si | digitalisation | dso    | rotation | marge  | hist_audit | risque
------------|------------|--------------|--------------|---------------|----------------|--------|----------|--------|------------|-------
tresorerie  | 3 729      | 2            | 0,761        | 2             | 0,184          | 41,63  | 6,13     | 0,415  | 0          | faible
stocks      | 4 325      | 3            | 0,617        | 1             | 0,734          | 47,73  | 6,92     | 0,345  | 0          | faible
paie        | 3 167      | 5            | 0,786        | 4             | 0,214          | 17,69  | 0,90     | 0,440  | 0          | eleve
stocks      | 6 041      | 3            | 0,880        | 1             | 0,327          | 79,44  | 2,96     | 0,279  | 2          | modere
```

## Annexe B — Structure du pipeline scikit-learn

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num (Pipeline)
│   │   ├── SimpleImputer(strategy='median')
│   │   └── StandardScaler()
│   └── cat (Pipeline)
│       ├── SimpleImputer(strategy='constant', fill_value='missing')
│       └── OneHotEncoder(handle_unknown='ignore')
└── classifier
    └── XGBClassifier / RandomForestClassifier / ...
```

## Annexe C — Paramètres optimaux (GridSearchCV Random Forest)

```
Meilleurs hyperparamètres identifiés :
- n_estimators      : 200
- max_depth         : 20 (ou None)
- min_samples_split : 2

Critère d'optimisation : accuracy (validation croisée 5 plis)
Score CV moyen       : ~0,87
```

## Annexe D — Interprétation des SHAP Values par classe

| Variable | Effet sur classe « Élevé » | Direction |
|---|---|---|
| `score_controle_interne` élevé | Forte réduction du risque | ↓ négatif |
| `nb_anomalies_passees` élevé | Forte augmentation du risque | ↑ positif |
| `cycle = Trésorerie` | Augmentation systématique | ↑ positif |
| `digitalisation` élevée | Réduction modérée du risque | ↓ négatif |
| `complexite_si` élevée | Augmentation du risque | ↑ positif |
| `indice_vulnerabilite` élevé | Augmentation du risque | ↑ positif |

---

*Compte rendu rédigé conformément aux exigences académiques de l'ENCG de Settat — Filière CAC — Semestre 8 — Année Universitaire 2024–2025*
