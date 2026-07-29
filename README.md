# Classification comics / manga sur AWS

Projet ESGI M1, module Terraform CI/CD.

Un pipeline qui prend des images rangées dans S3, les redimensionne, entraîne un
classifieur sur SageMaker, le déploie derrière un endpoint, prédit la classe des
images de test, écrit les résultats, puis supprime l'endpoint. Le tout est
orchestré par Step Functions, décrit en Terraform et déployé par GitHub Actions.

AWS plutôt qu'Azure : le sujet prévoit un bonus pour un fournisseur non vu en
cours.

## Résultat

2 000 images d'entraînement, 542 de test. Régression logistique sur 30
descripteurs de couleur.

- validation croisée à 5 plis : 0,71
- exactitude sur le jeu de test : 0,66
- une exécution complète : environ 13 minutes et 0,02 $

Le modèle penche vers « manga ». Les planches sont majoritairement en noir et
blanc là où les comics sont colorisés, et les comics eux-mêmes en noir et blanc
sont la principale source d'erreur.

## Le pipeline

```
PrepareDataset       liste les images, fabrique le run_id
ResizeImages         Map, 4 lots en parallèle, sortie en 64×64 PNG
TrainModel           Processing Job SageMaker -> model.tar.gz
CreateModel / CreateEndpointConfig / CreateEndpoint
WaitForEndpoint      boucle jusqu'à InService
RunInference         invoque l'endpoint, écrit parquet / csv / xlsx / json
DeleteEndpoint       atteint aussi depuis les chemins d'échec
```

## Déployer

Il faut Terraform ≥ 1.5, Python 3.13 et des identifiants AWS.

```bash
./tools/build_lambdas.sh
terraform -chdir=terraform init
terraform -chdir=terraform apply
```

`build_lambdas.sh` installe les dépendances des Lambdas pour Amazon Linux
x86_64, pas pour la machine locale. Il doit tourner avant tout `plan` ou
`apply`.

Lancer une exécution :

```bash
aws stepfunctions start-execution --region eu-central-1 --state-machine-arn arn:aws:states:eu-central-1:397068653698:stateMachine:ML_Pipeline --input '{}'
```

Les résultats arrivent dans `s3://projet-terraform/output/<run_id>/` : les
prédictions en Parquet, CSV et Excel, les métriques en JSON.

## CI/CD

`.github/workflows/terraform.yml` publie un `plan` en commentaire sur les pull
requests et fait l'`apply` sur `main`. Deux jobs d'analyse tournent en amont :
`controles` (compilation Python, `terraform fmt -check`) qui bloque l'apply, et
`securite` (tflint, checkov) qui ne bloque rien.

L'authentification passe par OIDC, il n'y a aucune clé AWS dans le dépôt ni dans
les secrets GitHub. Pour l'activer sur un autre compte : renseigner
`github_repository` dans `terraform/terraform.tfvars`, appliquer, puis déclarer
la sortie `github_actions_role_arn` en variable de dépôt `AWS_ROLE_ARN`.

## Organisation

```
.github/workflows/   le workflow Terraform
lambdas/             process_image (plan et resize), run_inference
scripts/             features.py, train.py, inference.py
statemachine/        la définition Step Functions
terraform/           main.tf, iam.tf, variables.tf, outputs.tf
tools/               build_lambdas.sh, smoke_test_inference.py
```

Le state est dans `s3://projet-terraform/terraform/`. Le bucket est déclaré en
`data` et non en `resource` : un `destroy` ne peut pas effacer le jeu de
données.

Le détail des choix techniques et des mesures est dans le dossier technique.
