# 🧠 PROJET TP Python EICNAM - Document Classifier
### Eloïs GUERIT
### 2025
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-1.14.0-orange.svg)
![Docker](https://img.shields.io/badge/docker-20.10.7-blue.svg)
![VSCode](https://img.shields.io/badge/VSCode-1.80.0-blue.svg)

Ce projet est une pipeline complete de **classification automatique de documents PDF**. Il utilise un modèle MLP exporté en **ONNX**, avec un **vectoriseur TF-IDF**, pour prédire la catégorie d’un document texte extrait depuis un PDF. Une fois classé, le fichier est renommé et déplacé dans le dossier approprié.

## 🚀 Fonctionnalités

- 📂 Surveillance automatique d’un dossier `deposit/`
- 🧾 Extraction de texte depuis les PDF
- 🧹 Nettoyage du texte
- 🤖 Prédiction de la catégorie via modèle ONNX
- 🗃️ Déplacement automatique dans le bon dossier (avec renommage `[index].pdf`)
- 🧼 Suppression automatique des fichiers non-PDF ou illisibles

---

## 🏗️ Structure du projet
```
project-root/
├── .devcontainer/ 
├── main.py # Script principal de surveillance et classification
├── modele-builder.ipynb # Notebook de conception et entraînement du modèle
├── modeles/
│ ├── tfidf.pkl 
│ └── clf.onnx
├── deposit/ # Où déposer les fichiers PDF
└── data/ # Contient un dossier par catégorie, avec les fichiers classés
├──── attestation hebergement/
├──── impot taxe foncier/
├──── impot sur revenus/
├──── bulletin de salaire/
└──── releve de compte bancaire/
```


---

## 🧪 Notebook : `modele-builder.ipynb`

Le fichier `modele-builder.ipynb` retrace l’intégralité du processus :

- Préparation des données
- Entraînement du `TfidfVectorizer`
- Entraînement d’un `MLPClassifier`
- Évaluation
- Export du pipeline au format ONNX
- Sauvegarde du vectoriseur avec `joblib`

Ce notebook est essentiel pour comprendre et reproduire le processus d'entraînement.

---

## ⚙️ Utilisation avec le DevContainer

Le projet contient une configuration **DevContainer** (Visual Studio Code + Docker) pour faciliter l’installation et garantir un environnement reproductible.

### Prérequis

- [Docker](https://www.docker.com/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Extension VS Code : **Remote - Containers**

### Étapes

1. Ouvre le projet dans VS Code
2. Clique sur : `Reopen in Container`
3. Attends que l’environnement se construise automatiquement

---

## ▶️ Exécution

Une fois dans le DevContainer ou dans un environnement Python avec les bonnes dépendances installées :

1. Lancer le script principal :

```bash
python main.py
```

2. Déposez vos fichiers PDF dans le dossier `deposit/`


Exemple :
```
cp deposit/mon_document.pdf deposit/
```

Le fichier sera automatiquement traité et déplacé dans l’un des dossiers suivants :



| Catégorie prédite         | Dossier cible                     |
| ------------------------- | --------------------------------- |
| Attestation hébergement   | `data/attestation hebergement/`   |
| Impôt / taxe foncière     | `data/impot taxe foncier/`        |
| Impôt sur le revenu       | `data/impot sur revenus/`         |
| Bulletin de salaire       | `data/bulletin de salaire/`       |
| Relevé de compte bancaire | `data/releve de compte bancaire/` |


## 🧼 Nettoyage


```
rm -rf deposit/* data/*/*.pdf
```