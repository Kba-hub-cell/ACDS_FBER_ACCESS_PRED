#  Prédiction de l'Accès FTTH au Togo



##  Description du Projet

Ce projet vise à prédire l'accès potentiel à la fibre optique (FTTH - Fiber to the Home) pour les ménages au Togo. En utilisant une combinaison de données socio-démographiques et de caractéristiques géospatiales MOSAIKS dérivées d'images satellites, nous développons des modèles de machine learning pour identifier les zones à fort potentiel de connectivité.

###  Objectifs

1. **Analyser** les facteurs influençant l'accès à Internet des ménages togolais
2. **Prédire** la probabilité d'accès FTTH pour chaque ménage
3. **Identifier** les zones prioritaires pour le déploiement de l'infrastructure
4. **Comprendre** les déterminants socio-économiques de la connectivité numérique

## 📁 Structure du Projet

```
ACDS_FBER_ACCESS_PRED/
│
├── 📂 Data/                      # Données du projet
│   ├── Data.csv                  # Dataset principal (~2.2 Go)
│   └── Métadonnées.csv           # Description des variables
│
├── 📂 notebooks/                 # Notebooks Jupyter
│   └── Fiber_Acces_pred.ipynb    # Notebook principal d'analyse
│
├── 📂 Output/                    # Résultats et exports
│   ├── best_model_*.joblib       # Modèle entraîné
│   ├── predictions_*.csv         # Prédictions
│   └── rapport_*.json            # Rapport de performance
│
├── 📄 environment.yml            # Environnement Conda
├── 📄 requirements.txt           # Dépendances pip (alternative)
├── 📖 README.md                  # Documentation


```

##  Installation

### Prérequis

- [Anaconda](https://www.anaconda.com/download) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (optionnel)

### Étape 1 : Cloner le projet

```bash
git clone https://github.com/Kba-hub-cell/ACDS_FBER_ACCESS_PRED.git
cd ACDS_FBER_ACCESS_PRED
```

### Étape 2 : Créer l'environnement Conda

```bash
# Créer l'environnement à partir du fichier environment.yml
conda env create -f environment.yml

# Activer l'environnement
conda activate FTTH_Togo
```

### Étape 3 : Lancer Jupyter Notebook

```bash
# Option 1 : Jupyter Notebook classique
jupyter notebook notebooks/Fiber_Acces_pred.ipynb

# Option 2 : JupyterLab
jupyter lab

# Option 3 : VS Code
# Ouvrir le notebook dans VS Code et sélectionner le kernel "FTTH_Togo"
```

##  Description des Données

### Dataset Principal (Data.csv)

| Catégorie                | Variables                               | Description                                    |
| ------------------------ | --------------------------------------- | ---------------------------------------------- |
| **Identifiants**         | ID, longitude, latitude                 | Localisation des ménages                       |
| **Socio-démographiques** | TypeLogmt, TAILLE_MENAGE, H08_Impute... | Caractéristiques des ménages (RGPH/INSEED)     |
| **Équipements**          | H17*\*, H18*_, H20\__, H21\_\*          | Possession d'équipements                       |
| **Connectivité**         | Connexion, BoxLabel                     | État de la connexion actuelle                  |
| **MOSAIKS**              | .1 à .3999                              | 4000 features géospatiales (images satellites) |
| **Target**               | Accès internet                          | Variable cible binaire (0/1)                   |

### Dimensions

- **30 558** observations (ménages)
- **4 002** variables
- **~2.2 Go** (format CSV)

##  Modèles Utilisés

| Modèle                  | Description                | Avantages                       |
| ----------------------- | -------------------------- | ------------------------------- |
| **Logistic Regression** | Modèle linéaire de base    | Interprétable, rapide           |
| **Random Forest**       | Ensemble bagging           | Robuste, feature importance     |
| **XGBoost**             | Gradient boosting          | Performance élevée              |
| **LightGBM**            | Gradient boosting optimisé | Rapide, efficace en mémoire     |
| **SVM**                 | Support Vector Machine     | Bon pour données complexes      |
| **MLP**                 | Réseau de neurones         | Capture relations non-linéaires |

##  Métriques d'Évaluation

- **AUC-ROC** : Capacité de discrimination du modèle
- **F1-Score** : Équilibre précision/rappel
- **Accuracy** : Taux de classification correcte
- **Precision/Recall** : Performance sur la classe positive

##  Interprétabilité

Le projet utilise **SHAP (SHapley Additive exPlanations)** pour :

- Identifier les features les plus importantes
- Comprendre l'impact de chaque variable
- Visualiser les interactions entre features

##  Optimisations de Performance

Le fichier de données étant volumineux (2.2 Go), plusieurs optimisations sont implémentées :

1. **Format Parquet** : Conversion automatique pour des chargements 5-10x plus rapides
2. **Types optimisés** : float32 au lieu de float64 pour les MOSAIKS
3. **Sélection de features** : Réduction des 4000 MOSAIKS aux plus pertinentes

##  Outputs

Après exécution, le dossier `Output/` contiendra :

| Fichier               | Description                              |
| --------------------- | ---------------------------------------- |
| `best_model_*.joblib` | Modèle entraîné prêt pour le déploiement |
| `scaler_*.joblib`     | Scaler pour normalisation                |
| `predictions_*.csv`   | Prédictions avec probabilités            |
| `rapport_*.json`      | Métriques de performance                 |

##  Contribution

Ce projet fait partie de l'initiative **African Citizen Data Scientist** visant à développer les compétences en science des données en Afrique.

### Auteurs

- Projet réalisé dans le cadre du programme ACDS par Alfred Kwami BIAM

##  Licence

Ce projet est développé à des fins éducatives et de recherche.

---

##  Dépannage

### L'environnement Conda ne se crée pas

```bash
# Mettre à jour Conda
conda update conda

# Réessayer la création
conda env create -f environment.yml
```

### Le kernel n'apparaît pas dans Jupyter/VS Code

```bash
# Activer l'environnement
conda activate FTTH_Togo

# Enregistrer le kernel
python -m ipykernel install --user --name=FTTH_Togo --display-name="Python (FTTH_Togo)"
```

### Problème de mémoire avec le fichier de données

Le notebook convertit automatiquement les données en format Parquet après le premier chargement, ce qui accélère les chargements suivants et réduit l'utilisation mémoire.

---

**Pour toute question ou collaboration, n'hésitez pas à me contacter.**
