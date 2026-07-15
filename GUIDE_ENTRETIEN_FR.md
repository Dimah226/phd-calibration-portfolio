# Guide de compréhension et préparation à l'entretien

## 1. Le problème étudié

Un classifieur produit souvent un score entre 0 et 1. Deux propriétés différentes
doivent être distinguées :

- **Discrimination** : les fraudes reçoivent-elles généralement un score plus élevé
  que les transactions normales ?
- **Calibration** : parmi les observations recevant une probabilité proche de 20 %,
  environ 20 % sont-elles réellement positives ?

Un modèle peut très bien classer les observations tout en produisant de mauvaises
probabilités. En fraude, cette distinction compte parce qu'un score sert non seulement
à classer des alertes, mais parfois aussi à estimer un risque et à prendre une décision.

## 2. Pourquoi le déséquilibre pose problème

Dans les données utilisées, environ 0,17 % des transactions sont frauduleuses. La
classe positive est donc extrêmement rare.

Le sous-échantillonnage consiste à retirer une partie des transactions normales afin
de faciliter l'apprentissage. Il modifie toutefois artificiellement la proportion de
fraudes dans les données d'entraînement. Un modèle entraîné sur cet échantillon apprend
des probabilités correspondant à cette nouvelle proportion, et non à la population
réelle.

La correction analytique utilisée dans le projet remet ces probabilités sur l'échelle
du taux de fraude réel. Si `beta` est la fraction de transactions normales conservées :

```text
p_population = p_sample / (p_sample + (1 - p_sample) / beta)
```

Cette correction suppose que les observations négatives ont été retirées aléatoirement.

## 3. Déroulement du pipeline

### Étape 1 — Données

La cellule de chargement du notebook cherche successivement :

1. `data/creditcard.csv` en local ;
2. la copie publique OpenML du jeu de données ;
3. des données synthétiques si le téléchargement échoue.

Le jeu est séparé en entraînement et test avec stratification, afin de conserver
approximativement la même proportion de fraudes dans les deux parties.

### Étape 2 — Modèle de référence

Une Random Forest est entraînée sans rééquilibrage. Le résultat important est que le
modèle est déjà relativement bien calibré globalement. L'expérience n'a donc pas
confirmé l'hypothèse initiale selon laquelle le modèle de référence serait clairement
surconfiant.

### Étape 3 — Calibration post-hoc

Deux méthodes sont comparées :

- **Platt scaling** : apprend une transformation sigmoïde des scores ;
- **régression isotone** : apprend une transformation monotone plus flexible.

Elles ne changent normalement pas beaucoup le classement des observations, mais elles
peuvent modifier la qualité probabiliste. Ici, Platt améliore l'ECE globale tout en
détériorant fortement la calibration du top 1 %. Une méthode ne gagne donc pas sur
tous les critères.

### Étape 4 — Sous-échantillonnage

La majorité négative est sous-échantillonnée. Les probabilités brutes deviennent très
mauvaises sur la population réelle. La correction de prior supprime l'essentiel de ce
biais, sans changer l'AUC-ROC : la correction transforme les probabilités, pas l'ordre
des scores.

### Étape 5 — Loss différentiable

Deux réseaux identiques sont entraînés :

- BCE/cross-entropy seule ;
- BCE + une approximation différentiable de l'ECE.

L'ECE ordinaire utilise des bins durs et n'est pas directement différentiable. La
fonction `soft_ece_loss` affecte chaque observation à plusieurs bins avec des poids
gaussiens. On peut alors calculer une approximation lisse de l'écart entre confiance
moyenne et fréquence observée, puis rétropropager son gradient.

Les deux réseaux sont entraînés sur les mêmes données sous-échantillonnées. La même
correction de prior leur est appliquée avant comparaison afin de ne pas confondre
l'effet de la loss avec celui du changement de prévalence.

## 4. Métriques à savoir expliquer

- **AUC-ROC** : probabilité qu'un positif choisi au hasard soit mieux classé qu'un
  négatif choisi au hasard. Elle peut paraître très bonne en cas de fort déséquilibre.
- **AUC-PR** : compromis précision-rappel ; souvent plus informative pour une classe
  positive rare.
- **Log-loss** : pénalise fortement les probabilités confiantes et fausses.
- **Brier score** : moyenne de `(probabilité - résultat)^2` ; plus petit est meilleur.
- **ECE** : moyenne pondérée des écarts calibration/confiance dans des bins de largeur
  égale.
- **ACE** : variante avec des bins contenant approximativement le même effectif.
- **Top-1% calibration error** : écart entre probabilité moyenne et fréquence réelle
  parmi le 1 % des alertes les mieux classées.

ECE et ACE dépendent du découpage en bins. Aucune de ces métriques ne doit être
présentée comme une vérité absolue.

## 5. Résultat principal

Après correction du prior, la loss soft-ECE améliore nettement l'erreur de calibration
dans le top 1 % des scores. Elle améliore très légèrement la log-loss et le Brier score,
mais détériore légèrement l'ECE globale, l'ACE et l'AUC-ROC.

La conclusion correcte n'est donc pas « ma loss résout la calibration ». La conclusion
est :

> Une loss de calibration peut déplacer le compromis entre calibration locale,
> calibration globale et discrimination. Son évaluation dépend de la métrique et de
> la zone opérationnelle considérée.

## 6. Limites à reconnaître en entretien

- Une seule séparation entraînement/test et une seule seed.
- Poids de la soft-ECE choisi de manière exploratoire.
- Pas de validation systématique des hyperparamètres.
- Expérience binaire alors que le doctorat vise le multiclasse.
- Métriques bin-based sensibles au choix des bins.
- Pas d'intervalles d'incertitude sur les résultats.
- La correction de prior suppose un sous-échantillonnage aléatoire de la classe négative.
- Ce projet illustre une question ; il ne propose pas une méthode scientifique nouvelle.

## 7. Questions probables en entretien

### Pourquoi ne pas se contenter de l'accuracy ou de l'AUC ?

Parce qu'elles évaluent la décision ou le classement, pas la signification probabiliste
des scores. Deux modèles ayant la même AUC peuvent avoir des probabilités très
différentes.

### Pourquoi l'ECE n'est-elle pas différentiable ?

Parce que l'affectation dure d'une observation à un bin change de manière discontinue
lorsque sa probabilité franchit une frontière. Le projet remplace cette affectation par
des poids continus.

### Pourquoi une loss de calibration peut-elle produire des prédictions constantes ?

Une prédiction presque constante proche du taux positif peut être globalement bien
calibrée tout en étant inutile pour discriminer les observations. La cross-entropy et la
loss de calibration doivent donc être équilibrées.

### Pourquoi appliquer la correction de prior aux deux MLP ?

Pour effectuer une comparaison équitable. Sans elle, l'erreur observée provient surtout
de la prévalence artificielle des données d'entraînement.

### Pourquoi le multiclasse est-il plus difficile ?

Il faut calibrer un vecteur de probabilités dont la somme vaut 1. Une bonne calibration
de la confiance maximale peut masquer une mauvaise calibration de certaines classes,
surtout les classes rares. Il existe plusieurs notions : calibration de confiance,
classwise, marginale ou calibration jointe.

### Quelle serait la prochaine expérience ?

Ajouter un jeu multiclasse public, répéter les expériences sur plusieurs seeds, réserver
un ensemble de validation pour la loss, et comparer des métriques de confiance et
classwise avec intervalles d'incertitude.

## 8. Explication des fichiers techniques

### Pourquoi un seul notebook ?

Le dépôt a été volontairement simplifié : les fonctions de chargement, métriques,
calibration et apprentissage sont définies directement dans
`calibration_walkthrough.ipynb`. Le notebook peut ainsi être lu et exécuté de haut en
bas sans dépendre de fichiers Python locaux.

### `.gitignore`

Ce fichier indique à Git ce qui ne doit pas être versionné :

- `__pycache__/` et `*.pyc` : fichiers temporaires créés par Python ;
- `.ipynb_checkpoints/` : sauvegardes automatiques de Jupyter ;
- `data/*.csv` : jeu brut volumineux, téléchargé séparément ;
- `.venv/` et `venv/` : environnements Python locaux contenant des milliers de fichiers.

Le `.gitignore` ne supprime rien. Il empêche seulement Git de proposer ces fichiers dans
les commits.
