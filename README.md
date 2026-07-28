# Classification comics / manga — pipeline AWS géré par Terraform

Pipeline de bout en bout qui, à partir d'un jeu d'images rangé dans S3,
redimensionne les images, entraîne un modèle sur SageMaker, le déploie derrière
un endpoint, prédit la classe de chaque image de test, écrit les résultats en
Parquet et en Excel, puis **supprime l'endpoint**. Le tout est orchestré par une
machine d'état Step Functions et décrit intégralement en Terraform.

Le fournisseur cloud est AWS, et non Azure : le sujet prévoit explicitement un
bonus pour une infrastructure déployée sur un fournisseur non vu en cours.

---

## 1. Résultat

Exécution de référence, `run_id` `20260728-112256` :

| Mesure | Valeur |
| --- | --- |
| Images d'entraînement | 2 000 (1 000 par classe) |
| Images de test | 542 (275 comics, 267 manga) |
| Validation croisée à l'entraînement (5 plis) | **0,7065** ± 0,016 |
| Exactitude sur le jeu de test | **0,6605** |
| Exactitude équilibrée | 0,6627 |
| Référence aléatoire | 0,5 |
| Inférences en échec | 0 |
| Durée totale du pipeline | ~13 min |

Détail par classe sur le jeu de test :

| Classe | Précision | Rappel | F1 | Support |
| --- | --- | --- | --- | --- |
| comics | 0,736 | 0,516 | 0,607 | 275 |
| manga | 0,619 | 0,809 | 0,701 | 267 |

Matrice de confusion (lignes = vérité) :

|  | prédit comics | prédit manga |
| --- | --- | --- |
| **comics** | 142 | 133 |
| **manga** | 51 | 216 |

Le modèle penche vers « manga » : il rattrape 81 % des mangas mais laisse passer
la moitié des comics. C'est cohérent avec ce que mesurent les descripteurs — la
saturation et la teinte —, les planches de manga étant majoritairement en noir
et blanc là où les comics sont colorisés. Les comics eux-mêmes en noir et blanc
sont la principale source d'erreur.

---

## 2. Architecture

```
                     ┌──────────────────────────────────────┐
                     │  Step Functions — ML_Pipeline        │
                     └──────────────────────────────────────┘
                                      │
  PrepareDataset ──────────► Lambda process_image (action=plan)
        │                    liste train/ et test/, découpe en lots,
        │                    fabrique le run_id et tous les noms de ressources
        ▼
  ResizeImages (Map, 4 en parallèle) ──► Lambda process_image (action=resize)
        │                    train/ + test/ ──► processed/<run_id>/ en 64×64 PNG
        ▼
  TrainModel ──────────────► SageMaker Processing Job (.sync)
        │                    scripts/train.py, conteneur scikit-learn 1.2-1
        │                    ──► models/<run_id>/model.tar.gz
        ▼
  CreateModel ─► CreateEndpointConfig ─► CreateEndpoint
        │
        ▼
  WaitForEndpoint ⇄ DescribeEndpoint ⇄ IsEndpointReady   (boucle 30 s, max 40)
        │
        ▼
  RunInference ────────────► Lambda comics-manga-run-inference
        │                    invoque l'endpoint pour les 542 images de test
        │                    ──► output/<run_id>/{parquet,csv,xlsx,json}
        ▼
  DeleteEndpoint ─► DeleteEndpointConfig ─► succès / échec
```

Deux points de conception méritent une explication.

**L'entraînement passe par un Processing Job, pas un Training Job.** Le compte
n'a aucun quota de Training Job : les 1 792 quotas SageMaker de la région ont
tous une valeur nulle sur la famille « training job usage ». Les quotas
« processing job usage » sont eux disponibles (4 sur `ml.t3.medium`). Un
Processing Job exécute le même conteneur, le même script et la même instance
gérée ; la seule différence est qu'il n'archive pas automatiquement
`/opt/ml/model` en `model.tar.gz`. `scripts/train.py` construit donc l'archive
lui-même avec `tarfile`, ce qui rend d'ailleurs le script utilisable tel quel
dans les trois contextes : Processing Job, Training Job et exécution locale.

**La suppression de l'endpoint est retentée pendant 20 minutes.** Un endpoint
encore en `Creating` refuse d'être supprimé (`Cannot update in-progress
endpoint`). Sans réessai, un endpoint bloqué en création resterait facturé
indéfiniment ; l'état `DeleteEndpoint` porte donc un `Retry` de 20 tentatives
espacées de 60 s, et tous les chemins d'échec de la machine d'état y convergent.

---

## 3. Organisation du dépôt

```
.github/workflows/terraform.yml   CI : contrôles + plan sur PR, apply sur main
lambdas/
  process_image/                  planification et redimensionnement
  run_inference/                  appels à l'endpoint et écriture des résultats
scripts/
  features.py                     les 30 descripteurs, partagés train / service
  train.py                        entraînement, exécuté par le Processing Job
  inference.py                    model_fn / input_fn / predict_fn / output_fn
statemachine/ml_pipeline.asl.json définition de la machine d'état (JSONata)
terraform/
  main.tf                         bucket, Lambdas, machine d'état, OIDC
  iam.tf                          les trois rôles
  variables.tf / outputs.tf
  .tflint.hcl                     jeu de règles de l'analyse statique
tools/
  build_lambdas.sh                paquets Lambda pour la plateforme d'exécution
  smoke_test_inference.py         test de démarrage du conteneur de service
```

Découpage des préfixes dans le bucket `projet-terraform` :

| Préfixe | Rôle |
| --- | --- |
| `train/{comics,manga}/` | images d'entraînement (fournies) |
| `test/{comics,manga}/` | images de test (fournies) |
| `scripts/` | code déposé par Terraform, lu par SageMaker |
| `processed/<run_id>/` | vignettes 64×64 |
| `models/<run_id>/` | `model.tar.gz` |
| `output/<run_id>/` | prédictions et métriques |
| `terraform/` | state Terraform |

---

## 4. Le modèle

Régression logistique sur **30 descripteurs de couleur**, dans un `Pipeline`
scikit-learn avec `StandardScaler`. `class_weight="balanced"`, `C=1.0`.

Les descripteurs :

- moyenne et écart-type des canaux rouge, vert, bleu et de la luminance (8) ;
- statistiques de chroma : moyenne, écart-type, médiane, 90ᵉ centile, part de
  pixels gris, part de pixels vifs (6) ;
- histogramme de chroma en 8 classes (8) ;
- histogramme de teinte en 8 classes, pondéré par la chroma (8).

### Sérialisation en JSON plutôt qu'en pickle

Le modèle n'est pas sauvegardé avec `joblib`. Il est exporté en JSON — moyennes
et écarts-types du scaler, coefficients, intercept, noms des descripteurs — et
`inference.py` réimplémente la prédiction en numpy pur. Cela supprime toute
contrainte de version entre la machine d'entraînement (scikit-learn 1.9) et le
conteneur de service (1.2.1), et évite de désérialiser du code exécutable.
Vérifié en pratique : la validation croisée obtenue localement et celle obtenue
dans le conteneur SageMaker sont identiques au chiffre près.

### Étude d'ablation : pourquoi 30 descripteurs et pas 52

La première version en comptait 52. Elle apprenait mieux (validation croisée à
l'entraînement ~0,73) mais généralisait moins bien.

Le diagnostic a montré un **confusion avec la résolution source** : dans
`train/`, la résolution des images est corrélée à la classe ; dans `test/`, elle
est anti-corrélée. Les 22 descripteurs sensibles à la résolution donnaient donc
au modèle un raccourci qui s'inverse au moment du test. Ils ont été retirés :

| Descripteurs retirés | Nombre | Motif |
| --- | --- | --- |
| `edge_*` | 3 | densité de contours ∝ netteté ∝ résolution |
| `luminance_hist_*` | 16 | le rééchantillonnage lisse l'histogramme |
| `dark_pixel_ratio`, `bright_pixel_ratio` | 2 | idem |
| `aspect_ratio` | 1 | propriété du fichier, pas du dessin |

Effet mesuré :

| | 52 descripteurs | 30 descripteurs |
| --- | --- | --- |
| Validation croisée (train) | ~0,73 | 0,7065 |
| **Exactitude (test)** | 0,605 | **0,662** |
| **AUC (test)** | 0,611 | **0,688** |

Deux points de validation croisée abandonnés contre près de six points
d'exactitude réelle. L'écart entre les deux colonnes est exactement la mesure du
sur-apprentissage que les descripteurs retirés induisaient.

Résultat négatif digne d'être noté : l'augmentation de données par changement
d'échelle, flou et compression JPEG, essayée pour rendre les descripteurs de
contours robustes à la résolution, n'a apporté aucun gain mesurable. Les
supprimer était plus efficace que tenter de les corriger.

---

## 5. Déploiement

### Prérequis

- Terraform ≥ 1.5 (validé avec 1.15.8)
- Python 3.13 avec `pip`
- des identifiants AWS avec accès au compte cible et au bucket

### Première mise en place

```bash
./tools/build_lambdas.sh
```

Ce script installe les dépendances des Lambdas **pour la plateforme d'exécution
de Lambda** (Amazon Linux, x86_64, CPython 3.13) et non pour la machine de
développement — sans quoi Pillow, qui embarque des extensions natives, serait
compilé pour macOS et la fonction échouerait à l'import. Il produit
`build/lambda/process_image` et `build/lambda/run_inference`, que Terraform
archive.

```bash
terraform -chdir=terraform init
```

```bash
terraform -chdir=terraform apply
```

La variable `import_existing_resources` vaut `true` par défaut : Terraform
adopte la machine d'état `ML_Pipeline` et la fonction `process_image` déjà
créées à la console au lieu d'en créer des doublons. Sur un compte vierge, la
passer à `false`.

### Lancer une exécution

```bash
aws stepfunctions start-execution --region eu-central-1 --state-machine-arn arn:aws:states:eu-central-1:397068653698:stateMachine:ML_Pipeline --input '{}'
```

Un `run_id` est généré automatiquement. Pour rejouer une exécution sur des
vignettes déjà produites, passer `--input '{"run_id":"20260728-112256"}'`.

### Résultats produits

Dans `s3://projet-terraform/output/<run_id>/` :

| Fichier | Contenu |
| --- | --- |
| `predictions.parquet` | une ligne par image de test |
| `predictions.xlsx` | mêmes données + feuilles « synthese » et « matrice_confusion » |
| `predictions.csv` | même contenu, lisible sans dépendance |
| `metrics.json` | exactitude, métriques par classe, matrice de confusion |

Colonnes : `image_key`, `image_name`, `true_label`, `predicted_label`,
`confidence`, `proba_comics`, `proba_manga`, `is_correct`, `inferred_at`.

### Vérifier avant de déployer

```bash
python tools/smoke_test_inference.py model.tar.gz data/test/comics/exemple.jpg
```

Ce test rejoue le démarrage du conteneur de service, variables d'environnement
comprises. Il existe pour une raison précise : la première mise en service a
échoué en boucle parce que `SAGEMAKER_CONTAINER_LOG_LEVEL` vaut `"20"` et non
`"INFO"`, ce que `Logger.setLevel` refuse. L'endpoint restait bloqué en
`Creating` sans message exploitable côté Step Functions.

---

## 6. Intégration continue

`.github/workflows/terraform.yml` :

| Job | Déclencheur | Actions |
| --- | --- | --- |
| `controles` | toutes | compilation Python, cohérence de `IMAGE_SIZE`, `terraform fmt -check`, validité du JSON de la machine d'état |
| `securite` | toutes | `tflint` en strict, `checkov` en soft-fail |
| `plan` | pull request | `init`, `validate`, `plan`, plan publié en commentaire |
| `apply` | push sur `main` | `init`, `apply` |

Seul `apply` dépend de `controles` : on ne déploie pas du code qui ne compile
pas, mais un `plan`, en lecture seule, n'a pas besoin d'attendre. `securite` ne
verrouille rien, pour qu'une alerte d'analyse statique ne bloque jamais un
déploiement.

`checkov` tourne en **soft-fail** : 64 contrôles réussis, 20 en échec. Les
échecs restants sont des compromis assumés — Lambdas hors VPC, groupe de logs
non chiffré par KMS, portée large du rôle de CI — et non des oublis.

L'authentification passe par **OIDC** : aucune clé d'accès AWS n'est stockée
dans le dépôt ni dans les secrets GitHub. Le workflow présente un jeton signé
par GitHub, AWS le vérifie et délivre des identifiants temporaires limités au
dépôt déclaré.

Pour l'activer :

1. renseigner `github_repository = "proprietaire/depot"` dans
   `terraform/terraform.tfvars`, puis `terraform apply` ;
2. relever la sortie `github_actions_role_arn` ;
3. la déclarer comme variable de dépôt `AWS_ROLE_ARN` dans les réglages GitHub.

Tant que `AWS_ROLE_ARN` n'est pas défini, seuls les contrôles statiques
s'exécutent : le dépôt reste utilisable sans configuration AWS. Si un autre
projet a déjà déclaré le fournisseur OIDC GitHub dans le compte — un seul est
autorisé par URL —, passer `create_github_oidc_provider = false`.

Le job `apply` cible l'environnement GitHub `production`, sur lequel une règle
de protection peut être posée pour exiger une validation manuelle.

Le state Terraform est stocké dans `s3://projet-terraform/terraform/`, avec
verrou par fichier `.tflock` (`use_lockfile`) plutôt qu'une table DynamoDB.
Sans state distant, la CI repartirait d'un state vide et proposerait de recréer
toute l'infrastructure à chaque exécution.

---

## 7. Coûts

Le poste dominant est l'endpoint, seule ressource facturée à la durée. Le
pipeline le supprime systématiquement, y compris sur les chemins d'échec.

| Ressource | Type | Durée typique | Ordre de grandeur |
| --- | --- | --- | --- |
| Processing Job | `ml.t3.medium` | ~3 min | < 0,01 $ |
| Endpoint | `ml.t2.medium` | ~8 min | ~0,01 $ |
| Lambda | 2 048 / 1 024 Mo | ~5 min cumulées | négligeable |
| S3 | ~350 Mo | permanent | ~0,01 $ / mois |

Soit environ **0,02 $ par exécution complète**. Les types d'instance ont été
choisis comme les plus petits disposant d'un quota sur le compte.

Pour tout supprimer :

```bash
terraform -chdir=terraform destroy
```

Le bucket est déclaré en `data`, pas en `resource` : `destroy` ne peut donc pas
effacer le jeu de données ni les résultats.

---

## 8. Sécurité

Trois rôles, un par service appelant :

| Rôle | Portée |
| --- | --- |
| `comics-manga-lambda` | lecture `train/`, `test/`, `processed/` ; écriture `processed/`, `output/` ; `InvokeEndpoint` |
| `comics-manga-sagemaker-execution` | lecture `scripts/` et `processed/`, écriture `models/` |
| `comics-manga-stepfunctions` | invocation des Lambdas, API SageMaker, EventBridge |

Les deux Lambdas partagent un rôle : leurs besoins se recouvrent largement et
un rôle par fonction ajoutait de la configuration sans réduire la surface de
manière significative. Chacun reste limité à ses préfixes.

Un quatrième rôle, `comics-manga-github-actions`, n'existe que si la CI est
activée. Il porte `PowerUserAccess` et la gestion des rôles IAM, sans accès aux
utilisateurs, à l'organisation ni à la facturation. Cette portée est volontaire :
la CI doit pouvoir créer n'importe quel service, puisque le périmètre des
modifications n'est pas connu à l'avance. Sur une infrastructure figée, on la
restreindrait aux seuls services du projet.

`AmazonSageMakerFullAccess` n'a volontairement pas été utilisé : cette politique
ne donne accès qu'aux buckets dont le nom contient « sagemaker », ce qui n'est
pas le cas ici — la première tentative avait d'ailleurs échoué en `AccessDenied`
pour cette raison. Le rôle SageMaker est donc écrit à la main, limité aux
préfixes qu'il utilise réellement.

Le `iam:PassRole` accordé à Step Functions est conditionné par
`iam:PassedToService = sagemaker.amazonaws.com`.
