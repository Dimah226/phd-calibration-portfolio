# La calibration des modèles, de zéro jusqu'au projet

Ce cours part de l'idée la plus simple possible et arrive progressivement au code du
projet. Lis-le dans l'ordre. L'objectif n'est pas de mémoriser des définitions, mais de
pouvoir expliquer avec tes propres mots ce que fait chaque expérience.

---

# Partie I — Les bases

## 1. Que fait un modèle de classification ?

Un modèle de classification reçoit des informations `X` et doit prédire une catégorie
`Y`.

Dans notre projet :

- `X` = caractéristiques d'une transaction bancaire ;
- `Y = 0` = transaction normale ;
- `Y = 1` = fraude.

Le modèle ne renvoie pas nécessairement directement `0` ou `1`. Il produit souvent un
nombre, par exemple :

```text
Transaction A -> 0,02
Transaction B -> 0,70
Transaction C -> 0,95
```

On appelle souvent ce nombre un **score**. La question fondamentale est : peut-on
réellement interpréter `0,70` comme « 70 % de probabilité de fraude » ?

La réponse est : pas automatiquement.

## 2. Score, classement, décision et probabilité

Ces quatre objets sont différents.

### Le score

Un nombre produit par le modèle. Un score plus grand signifie généralement que le
modèle considère la fraude plus plausible.

### Le classement

On trie les transactions du score le plus élevé au plus faible. C'est utile lorsqu'une
équipe ne peut examiner que les 100 alertes les plus suspectes.

### La décision

On choisit un seuil :

```text
si score >= 0,50 : prédire fraude
sinon             : prédire normal
```

Changer le seuil change les décisions, mais pas les scores eux-mêmes.

### La probabilité

Une probabilité de `0,70` fait une affirmation beaucoup plus forte qu'un simple score :
parmi beaucoup de cas comparables auxquels le modèle attribue environ `0,70`, environ
70 % devraient réellement être positifs.

La calibration étudie cette correspondance.

## 3. Définition intuitive de la calibration

Imaginons 100 transactions auxquelles le modèle attribue une probabilité proche de
`0,20`.

- Si environ 20 sont frauduleuses, le modèle est bien calibré autour de `0,20`.
- Si 60 sont frauduleuses, le modèle est **sous-confiant** : il annonce 20 % alors que
  le risque observé est proche de 60 %.
- Si seulement 5 sont frauduleuses, le modèle est **surconfiant** : il annonce 20 %
  alors que le risque observé est proche de 5 %.

Mathématiquement, une calibration parfaite signifie approximativement :

```text
P(Y = 1 | probabilité prédite = p) = p
```

Traduction : sachant que le modèle annonce `p`, la fréquence réelle de positifs doit
être `p`.

## 4. Un exemple numérique complet

Supposons dix observations :

| Observation | Probabilité prédite | Résultat réel |
|---|---:|---:|
| 1 | 0,10 | 0 |
| 2 | 0,10 | 0 |
| 3 | 0,10 | 0 |
| 4 | 0,10 | 1 |
| 5 | 0,40 | 0 |
| 6 | 0,40 | 1 |
| 7 | 0,40 | 0 |
| 8 | 0,80 | 1 |
| 9 | 0,80 | 1 |
| 10 | 0,80 | 1 |

Pour le groupe à `0,10`, la fréquence réelle est `1/4 = 0,25`. Le modèle sous-estime
le risque dans ce groupe.

Pour le groupe à `0,40`, la fréquence réelle est `1/3 = 0,33`. Le modèle est assez
proche.

Pour le groupe à `0,80`, la fréquence réelle est `3/3 = 1,00`. Le modèle sous-estime
encore le risque.

La calibration ne demande pas que chaque prédiction soit correcte. Une observation
ayant une probabilité de 20 % peut parfaitement être positive. Elle demande que les
fréquences soient correctes sur des groupes d'observations comparables.

---

# Partie II — Calibration et performance prédictive

## 5. Discrimination et calibration

La **discrimination** mesure la capacité à placer les positifs au-dessus des négatifs.
La **calibration** mesure la qualité de l'échelle probabiliste.

Prenons deux modèles :

```text
                Cas normal   Cas frauduleux
Modèle A           0,10          0,20
Modèle B           0,40          0,80
```

Les deux classent correctement la fraude au-dessus du cas normal. Leur discrimination
est identique sur ces deux observations. Mais leurs probabilités sont très différentes.

On peut même transformer tous les scores par une fonction croissante : le classement
ne change pas, donc l'AUC-ROC ne change pas, mais la calibration peut changer fortement.

Retenir :

> L'AUC évalue l'ordre des scores ; la calibration évalue leur signification.

## 6. Accuracy, précision et rappel

### Accuracy

```text
nombre de bonnes prédictions / nombre total d'observations
```

Avec 0,17 % de fraudes, un modèle qui prédit toujours « normal » atteint environ
99,83 % d'accuracy. Il est pourtant inutile.

### Précision

Parmi les alertes déclarées positives, quelle proportion est réellement frauduleuse ?

```text
précision = vrais positifs / (vrais positifs + faux positifs)
```

### Rappel

Parmi toutes les fraudes réelles, quelle proportion le modèle détecte-t-il ?

```text
rappel = vrais positifs / (vrais positifs + faux négatifs)
```

Ces métriques évaluent des décisions prises à un seuil. Elles ne disent toujours pas si
une valeur `0,70` signifie réellement 70 %.

## 7. Comprendre l'AUC depuis le début

Le sigle **AUC** signifie `Area Under the Curve`, c'est-à-dire « aire sous la courbe ».
Mais avant de comprendre l'aire, il faut comprendre pourquoi on construit une courbe.

### 7.1 Le problème du seuil

Le modèle produit un score. Pour transformer ce score en décision, on choisit un seuil.

Exemple :

| Transaction | Score | Réalité |
|---|---:|---|
| A | 0,90 | fraude |
| B | 0,70 | normale |
| C | 0,60 | fraude |
| D | 0,10 | normale |

Avec un seuil de `0,80` :

```text
A : 0,90 >= 0,80 -> alerte correcte
B : 0,70 <  0,80 -> rejet correct
C : 0,60 <  0,80 -> fraude manquée
D : 0,10 <  0,80 -> rejet correct
```

Le modèle trouve une fraude sur deux.

Avec un seuil de `0,50` :

```text
A -> alerte correcte
B -> fausse alerte
C -> alerte correcte
D -> rejet correct
```

Le modèle trouve maintenant les deux fraudes, mais crée une fausse alerte.

Il n'existe donc pas une seule performance du modèle : elle dépend du seuil. Les courbes
ROC et PR observent ce qui se passe lorsque l'on fait varier le seuil de 1 jusqu'à 0.

### 7.2 Vrais et faux positifs

Pour chaque seuil, on calcule notamment :

```text
TPR = rappel = fraudes détectées / toutes les fraudes
FPR = transactions normales déclarées frauduleuses / toutes les normales
```

`TPR` signifie `True Positive Rate`, taux de vrais positifs.

`FPR` signifie `False Positive Rate`, taux de faux positifs.

Un bon modèle cherche un TPR élevé et un FPR faible : détecter beaucoup de fraudes sans
déclencher trop de fausses alertes.

### 7.3 Construction de la courbe ROC

La courbe ROC place, pour chaque seuil :

```text
axe horizontal x = FPR
axe vertical   y = TPR
```

Lorsque le seuil est très élevé, presque aucune transaction n'est déclarée frauduleuse :

```text
TPR proche de 0
FPR proche de 0
```

Lorsque le seuil devient très faible, presque tout est déclaré frauduleux :

```text
TPR proche de 1
FPR proche de 1
```

Entre les deux, chaque seuil produit un point. Relier ces points donne la courbe ROC.

Une bonne courbe monte rapidement vers le coin supérieur gauche : beaucoup de fraudes
détectées pour peu de fausses alertes.

### 7.4 Que signifie AUC-ROC ?

L'**AUC-ROC** est l'aire située sous la courbe ROC.

Elle varie généralement entre `0` et `1` :

- `1,00` : classement parfait ; toutes les fraudes sont au-dessus de toutes les normales ;
- `0,50` : classement comparable au hasard ;
- moins de `0,50` : classement inversé ; retourner les scores ferait mieux.

L'interprétation la plus simple est :

> Si je choisis au hasard une fraude et une transaction normale, l'AUC-ROC est la
> probabilité que le modèle donne le score le plus élevé à la fraude.

Dans notre petit exemple, les scores des fraudes sont `0,90` et `0,60`. Les scores des
normales sont `0,70` et `0,10`.

On compare toutes les paires fraude/normale :

```text
fraude 0,90 contre normale 0,70 -> bon ordre
fraude 0,90 contre normale 0,10 -> bon ordre
fraude 0,60 contre normale 0,70 -> mauvais ordre
fraude 0,60 contre normale 0,10 -> bon ordre
```

Trois comparaisons sur quatre sont correctes :

```text
AUC-ROC = 3/4 = 0,75
```

Cette méthode par paires donne la même interprétation que l'aire sous la courbe.

### 7.5 Ce que l'AUC-ROC ne dit pas

Supposons deux modèles :

```text
                Fraude   Normale
Modèle 1          0,90      0,80
Modèle 2          0,20      0,10
```

Les deux placent la fraude au-dessus de la normale. Leur classement est donc identique
sur cette paire, malgré des probabilités complètement différentes.

L'AUC-ROC ne dit pas :

- si `0,90` signifie réellement 90 % ;
- quel seuil sera utilisé en production ;
- combien d'alertes seront fausses à ce seuil ;
- si le modèle est calibré.

Elle mesure le **classement**.

### 7.6 Pourquoi l'AUC-ROC peut être flatteuse en fraude

Imaginons :

```text
100 fraudes
100 000 transactions normales
```

Un taux de faux positifs de seulement 1 % semble faible :

```text
FPR = 1 %
```

Mais 1 % de 100 000 donne :

```text
1 000 fausses alertes
```

Si le modèle détecte 80 fraudes, l'équipe reçoit :

```text
80 vraies alertes
1 000 fausses alertes
```

La ROC retient surtout que le FPR vaut seulement 1 %. Pour l'équipe opérationnelle,
1 000 faux dossiers sont pourtant considérables. C'est une raison de regarder aussi la
précision et la courbe PR.

### 7.7 Précision et rappel

Rappel :

```text
rappel = fraudes détectées / toutes les fraudes
```

Dans l'exemple précédent :

```text
rappel = 80 / 100 = 80 %
```

Précision :

```text
précision = vraies alertes / toutes les alertes
```

Donc :

```text
précision = 80 / (80 + 1 000) = 7,4 % environ
```

Autrement dit, moins de 8 alertes sur 100 sont réellement frauduleuses.

### 7.8 Construction de la courbe précision-rappel

Pour chaque seuil, la courbe PR place :

```text
axe horizontal x = rappel
axe vertical   y = précision
```

Lorsque l'on abaisse le seuil :

- on détecte généralement plus de fraudes, donc le rappel augmente ;
- on ajoute aussi des fausses alertes, donc la précision peut diminuer.

La courbe montre ce compromis pour tous les seuils.

### 7.9 Que signifie AUC-PR ?

L'**AUC-PR** résume l'aire sous la courbe précision-rappel. Dans scikit-learn, le projet
utilise `average_precision_score`, une synthèse en escaliers de cette relation.

Plus cette valeur est élevée, plus le modèle maintient une bonne précision lorsqu'il
cherche à augmenter le rappel.

Attention : contrairement à l'AUC-ROC, la référence aléatoire de l'AUC-PR n'est pas
toujours `0,50`. Elle dépend de la prévalence de la classe positive.

Si seulement 0,17 % des transactions sont frauduleuses, un classement aléatoire a une
précision moyenne proche de :

```text
0,17 % = 0,0017
```

Une AUC-PR de `0,80` est donc extrêmement supérieure au hasard sur ce dataset.

### 7.10 Différence essentielle entre AUC-ROC et AUC-PR

| Mesure | Ce qu'elle regarde principalement | Sensible au déséquilibre |
|---|---|---|
| AUC-ROC | Classement positif contre négatif via TPR/FPR | Peut rester flatteuse avec beaucoup de négatifs |
| AUC-PR | Qualité des alertes positives via précision/rappel | Plus révélatrice pour une classe positive rare |

Dans le projet :

- l'AUC-ROC dit si les fraudes sont globalement placées avant les normales ;
- l'AUC-PR dit si les premières alertes restent pertinentes lorsque l'on cherche à
  récupérer davantage de fraudes ;
- les métriques de calibration disent si les valeurs numériques des probabilités sont
  fiables.

### 7.11 Pourquoi la correction de prior ne change pas l'AUC

La correction d'undersampling transforme les probabilités avec une fonction monotone.

Exemple :

```text
avant correction : A = 0,90 ; B = 0,60 ; C = 0,20
après correction : A = 0,30 ; B = 0,08 ; C = 0,01
```

Les valeurs changent, mais l'ordre reste :

```text
A > B > C
```

Comme les AUC dépendent du classement, elles restent identiques. La
calibration, elle, peut s'améliorer fortement parce que l'échelle probabiliste change.

### 7.12 Résumé en une phrase pour l'entretien

> L'AUC-ROC mesure la qualité globale du classement entre positifs et négatifs, tandis
> que l'AUC-PR mesure plus directement la qualité de la détection de la classe rare ;
> aucune des deux ne garantit que les scores soient des probabilités calibrées.

---

# Partie III — Mesurer la calibration

## 8. Diagramme de fiabilité

On découpe les probabilités en intervalles appelés **bins** :

```text
[0,00 ; 0,10[
[0,10 ; 0,20[
...
[0,90 ; 1,00]
```

Pour chaque bin, on calcule :

- la probabilité prédite moyenne ;
- la fréquence réelle moyenne de positifs.

On place ensuite un point :

```text
x = confiance moyenne
y = fréquence réelle
```

Un modèle parfaitement calibré suit la diagonale `y = x`.

- Point au-dessus de la diagonale : risque réel supérieur au risque annoncé,
  donc modèle sous-confiant.
- Point sous la diagonale : risque réel inférieur au risque annoncé,
  donc modèle surconfiant.

Dans les graphiques du projet, les gros marqueurs correspondent aux bins contenant le
plus d'observations.

## 9. Expected Calibration Error — ECE

Pour chaque bin `b`, on calcule :

```text
écart_b = |fréquence réelle_b - probabilité moyenne_b|
```

Puis on fait une moyenne pondérée par l'effectif de chaque bin :

```text
ECE = somme(effectif_b / effectif_total * écart_b)
```

Une ECE faible est généralement meilleure.

### Limite importante

L'ECE dépend :

- du nombre de bins ;
- de leurs frontières ;
- de la concentration des probabilités.

Avec énormément de transactions normales ayant des probabilités proches de zéro, ces
observations dominent la moyenne. L'ECE peut donc être faible même si les alertes les
plus importantes sont mal calibrées.

## 10. Adaptive Calibration Error — ACE

L'ECE classique utilise des bins de même largeur. L'ACE du projet trie les scores puis
construit des bins contenant approximativement le même nombre d'observations.

Cette méthode donne davantage de résolution dans les zones où les scores sont très
concentrés. Elle possède aussi ses propres limites. Le projet rapporte ECE et ACE pour
montrer que la conclusion peut dépendre du découpage.

## 11. Erreur de calibration du top 1 %

Une équipe de fraude ne regarde souvent qu'une petite file d'alertes. Le projet prend le
1 % des transactions ayant les scores les plus élevés et calcule :

```text
|fréquence de fraude dans le top 1 % - probabilité moyenne dans le top 1 %|
```

Cette métrique est plus opérationnelle, mais elle ne mesure qu'une zone des scores.
Elle ne remplace pas les métriques globales.

## 12. Brier score

Pour chaque observation :

```text
(probabilité prédite - résultat réel)^2
```

Puis on calcule la moyenne.

Exemples :

- prédire `0,90` lorsque `Y = 1` donne `(0,90 - 1)^2 = 0,01` ;
- prédire `0,90` lorsque `Y = 0` donne `(0,90 - 0)^2 = 0,81`.

Le Brier score pénalise donc les mauvaises probabilités. Plus petit est meilleur. Il
mélange cependant calibration et capacité de discrimination.

## 13. Log-loss

Pour une observation binaire :

```text
log-loss = -[y log(p) + (1-y) log(1-p)]
```

Elle pénalise très fortement une prédiction fausse et très confiante. Prédire une
probabilité presque nulle pour un positif réel coûte énormément.

La cross-entropy utilisée pour entraîner le réseau est cette même log-loss binaire.

---

# Partie IV — Pourquoi l'undersampling casse les probabilités

## 14. Le déséquilibre des classes

Le jeu réel contient environ :

```text
99,83 % de transactions normales
 0,17 % de fraudes
```

Si l'on conserve toutes les fraudes mais seulement une petite partie des transactions
normales, la proportion de fraudes devient artificiellement beaucoup plus élevée dans
l'échantillon d'entraînement.

Le modèle apprend alors la probabilité dans ce monde artificiel.

### 14.1 Une analogie avec une enquête

Supposons qu'une ville contienne :

```text
990 personnes sans emploi
 10 personnes avec emploi
```

Dans la ville réelle, le taux d'emploi est :

```text
10 / 1 000 = 1 %
```

Un enquêteur trouve qu'il y a trop de personnes sans emploi dans son fichier. Il décide
de conserver les 10 personnes avec emploi, mais seulement 10 personnes sans emploi.

Son échantillon contient maintenant :

```text
10 personnes avec emploi
10 personnes sans emploi
```

Dans ce fichier, le taux d'emploi vaut 50 %. Pourtant, la ville n'a pas changé : son
vrai taux reste 1 %.

Si un modèle apprend seulement à partir du fichier équilibré, il vit dans un monde où
l'événement positif semble cinquante fois plus fréquent qu'en réalité.

Le même problème apparaît lorsqu'on conserve toutes les fraudes mais qu'on retire une
grande partie des transactions normales.

### 14.2 Ce que l'undersampling conserve et ce qu'il modifie

Si les transactions normales sont retirées **aléatoirement**, l'undersampling peut
conserver beaucoup d'information sur les différences entre profils frauduleux et
normaux. Il peut donc conserver un classement utile.

Mais il modifie la proportion de base :

```text
P(fraude)
```

Cette proportion de base est aussi appelée **prior** ou **prévalence**.

Le mot « correction du prior » signifie donc : corriger les probabilités pour tenir
compte du fait que le taux de fraude vu pendant l'entraînement n'était pas le vrai taux
de la population.

## 15. Exemple intuitif

Supposons une population de 10 000 transactions :

```text
20 fraudes
9 980 normales
```

On conserve toutes les fraudes, mais seulement 100 normales :

```text
20 fraudes
100 normales
```

Dans l'échantillon, la fraude représente maintenant `20/120 = 16,7 %`, alors que dans
la population elle représente `20/10 000 = 0,2 %`.

Même si le classement appris est utile, les probabilités brutes sont beaucoup trop
élevées pour la population réelle.

### 15.1 Pourquoi une même transaction change de probabilité

Imaginons un type de transaction particulier : achat nocturne, montant inhabituel,
nouveau commerçant.

Dans la population réelle, on trouve parmi les transactions de ce profil :

```text
10 fraudes
990 transactions normales
```

Sa probabilité réelle de fraude est :

```text
10 / (10 + 990) = 1 %
```

Maintenant, on garde toutes les fraudes mais seulement 10 % des transactions normales.
Parmi les 990 normales de ce profil, il n'en reste que 99 environ :

```text
10 fraudes
99 transactions normales conservées
```

Dans l'échantillon sous-échantillonné, le même profil semble avoir une probabilité :

```text
10 / (10 + 99) = 9,17 % environ
```

Le modèle peut donc apprendre environ 9,17 %, alors que le vrai risque dans la
population est 1 %.

Le modèle n'est pas forcément devenu mauvais pour classer les profils. Il répond
simplement à la mauvaise question :

```text
« Quelle est la probabilité dans mon échantillon artificiel ? »
```

alors que nous voulons :

```text
« Quelle est la probabilité dans la population réelle ? »
```

## 16. Correction du prior

### 16.1 Les symboles

On note :

```text
p          = vraie probabilité dans la population
p_sampled  = probabilité apprise dans l'échantillon sous-échantillonné
beta       = fraction des transactions normales conservées
```

Exemples de `beta` :

```text
beta = 1,00 -> 100 % des normales conservées ; aucun undersampling
beta = 0,50 ->  50 % des normales conservées
beta = 0,10 ->  10 % des normales conservées
beta = 0,01 ->   1 % des normales conservées
```

Plus `beta` est petit, plus on a supprimé de transactions normales, et plus la
probabilité brute apprise est artificiellement gonflée.

### 16.2 Comment `beta` est calculé dans le projet

Dans la fonction `undersample` du notebook, `beta` est la fraction des négatifs conservés :

```python
beta = n_keep / len(neg_idx)
```

Supposons :

```text
200 000 transactions normales au départ
  2 000 transactions normales conservées
```

Alors :

```text
beta = 2 000 / 200 000 = 0,01
```

Nous n'avons gardé que 1 % des négatifs.

### 16.3 La formule utilisée

La correction est :

```python
return p_sampled / (p_sampled + (1 - p_sampled) / beta)
```

Écriture mathématique équivalente :

```text
                  p_sampled
p = --------------------------------------
      p_sampled + (1 - p_sampled) / beta
```

### 16.4 Calcul numérique complet

Supposons que :

```text
p_sampled = 0,20
beta      = 0,01
```

Le modèle entraîné sur l'échantillon artificiel annonce 20 %, mais seulement 1 % des
transactions normales avaient été conservées.

Calcul :

```text
1 - p_sampled = 1 - 0,20 = 0,80

(1 - p_sampled) / beta = 0,80 / 0,01 = 80

dénominateur = 0,20 + 80 = 80,20

p = 0,20 / 80,20 = 0,00249 environ
```

Donc :

```text
probabilité dans l'échantillon : 20 %
probabilité corrigée réelle    : 0,249 % environ
```

La forte baisse est logique : dans le fichier d'entraînement, nous avions retiré 99 %
des transactions normales, ce qui rendait artificiellement la fraude beaucoup plus
fréquente.

### 16.5 Vérification avec l'exemple à 1 %

Reprenons le profil contenant réellement :

```text
10 fraudes
990 normales
```

Après conservation de 10 % des normales :

```text
10 fraudes
99 normales
```

Nous avions trouvé :

```text
p_sampled = 10 / 109 = 0,09174
beta = 0,10
```

Appliquons la correction :

```text
p = 0,09174 / [0,09174 + (1 - 0,09174)/0,10]

p = 0,09174 / [0,09174 + 9,0826]

p = 0,09174 / 9,1743

p = 0,01 = 1 %
```

Nous retrouvons bien la probabilité réelle.

### 16.6 D'où vient la formule ?

Avant undersampling, pour un profil `x`, imaginons :

```text
nombre relatif de positifs = a
nombre relatif de négatifs = b
```

La vraie probabilité est :

```text
p = a / (a + b)
```

Après undersampling, on garde tous les positifs mais seulement une fraction `beta` des
négatifs :

```text
positifs conservés = a
négatifs conservés = beta * b
```

La probabilité dans l'échantillon devient :

```text
p_sampled = a / (a + beta * b)
```

La formule de correction consiste simplement à résoudre cette relation afin de
retrouver `a/(a+b)`. Elle ne « devine » donc pas la vraie probabilité : elle annule
mathématiquement la modification connue que nous avons nous-mêmes appliquée aux
négatifs.

### 16.7 Une autre lecture avec les odds

Les **odds** d'une probabilité `p` sont :

```text
odds = p / (1-p)
```

Exemple : `p = 0,20` donne :

```text
odds = 0,20 / 0,80 = 0,25
```

Cela signifie environ 1 positif pour 4 négatifs.

Lorsque l'on ne conserve qu'une fraction `beta` des négatifs, les odds observées dans
l'échantillon sont multipliées par `1/beta`, car le dénominateur négatif a été réduit.

Pour revenir à la population :

```text
odds_population = beta * odds_sampled
```

Puis on reconvertit les odds en probabilité :

```text
p = odds / (1 + odds)
```

Cette écriture conduit à la même formule.

### 16.8 Pourquoi la correction conserve le classement

Prenons trois probabilités brutes :

```text
A = 0,80
B = 0,40
C = 0,10
```

Après correction, elles peuvent devenir :

```text
A = 0,038
B = 0,0066
C = 0,0011
```

Les valeurs ont fortement baissé, mais :

```text
A > B > C
```

La fonction de correction est monotone croissante. Elle change l'échelle, pas l'ordre.
C'est pourquoi AUC-ROC et AUC-PR restent identiques avant et après correction dans les
résultats du projet.

### 16.9 Hypothèses nécessaires

La correction n'est pas magique. Elle est valable si :

1. tous les positifs sont conservés, ou si leur taux de conservation est connu ;
2. chaque négatif a la même probabilité `beta` d'être conservé ;
3. le sous-échantillonnage est aléatoire à l'intérieur de la classe négative ;
4. le principal changement entre échantillon et population vient de ce mécanisme ;
5. le modèle estime raisonnablement la probabilité dans l'échantillon sous-échantillonné.

Si l'on retire surtout certains types de négatifs, par exemple uniquement les petites
transactions, une simple correction globale de prior ne suffit plus : la distribution
des caractéristiques a également changé.

### 16.10 Ce que la correction peut et ne peut pas réparer

Elle peut réparer :

- le gonflement des probabilités causé par un undersampling aléatoire connu ;
- l'échelle probabiliste liée au changement de prévalence.

Elle ne peut pas automatiquement réparer :

- un mauvais classement ;
- des variables insuffisantes ;
- un modèle mal entraîné ;
- un changement de comportement des fraudeurs ;
- un undersampling dépendant des caractéristiques ;
- toutes les autres formes de mauvaise calibration.

### 16.11 Pourquoi le modèle corrigé reste moins bon que le modèle initial

Dans les résultats, la correction améliore énormément le modèle undersampled, mais elle
ne retrouve pas complètement la qualité de la Random Forest entraînée sur toutes les
données.

C'est normal : l'undersampling a supprimé beaucoup d'observations normales. La
correction restaure le prior, mais elle ne recrée pas l'information perdue pendant
l'entraînement. Le modèle peut donc avoir appris une frontière ou un classement moins
précis.

### 16.12 Résumé pour l'oral

> L'undersampling rend artificiellement la fraude plus fréquente dans les données
> d'entraînement. Le modèle apprend alors des probabilités correspondant à cet
> échantillon. Comme je connais la fraction de négatifs conservés, j'utilise une
> correction analytique pour replacer les scores sur l'échelle de prévalence de la
> population réelle. Cette correction change les probabilités, mais pas leur ordre.

---

# Partie V — Calibration post-hoc

## 17. Principe général

Le mot **post-hoc** signifie « après coup ».

La calibration post-hoc suit deux apprentissages successifs :

1. on entraîne le classifieur pour produire des scores ;
2. sans modifier ce classifieur, on apprend une fonction qui transforme ses scores en
   probabilités plus fiables.

```text
score brut -> fonction de calibration -> probabilité corrigée
```

Exemple :

```text
score brut du modèle       = 0,80
probabilité après calibration = 0,55
```

La fonction de calibration a appris que, dans les données disponibles, les cas ayant
un score brut proche de `0,80` ne sont positifs qu'environ 55 % du temps.

### 17.1 Le modèle de base ne change pas

Supposons une Random Forest déjà entraînée. La calibration post-hoc ne modifie pas ses
arbres. Elle observe seulement :

```text
scores produits par la Random Forest
résultats réels correspondants
```

Puis elle apprend une correspondance :

```text
score 0,20 -> probabilité corrigée 0,05
score 0,50 -> probabilité corrigée 0,30
score 0,80 -> probabilité corrigée 0,70
```

### 17.2 Pourquoi ne pas calibrer sur les données d'entraînement ?

Un modèle paraît généralement meilleur sur les observations qui ont servi à
l'entraîner. Ses scores peuvent y être trop optimistes.

Si la fonction de calibration apprend sur ces mêmes scores, elle apprend à corriger le
comportement du modèle sur des données déjà vues, pas nécessairement sur de nouvelles
données.

Il faut idéalement trois rôles distincts :

```text
Entraînement : ajuster le classifieur
Calibration  : ajuster la transformation des scores
Test         : évaluer l'ensemble final
```

Exemple avec 100 000 observations :

```text
60 000 -> entraînement du modèle
20 000 -> apprentissage de la calibration
20 000 -> évaluation finale
```

Le test ne doit servir ni à entraîner le modèle ni à choisir la transformation. Sinon,
on crée une **fuite de données** ou `data leakage`.

### 17.3 Que faire si l'on ne veut pas sacrifier un grand ensemble de calibration ?

On peut utiliser la validation croisée.

Avec trois folds :

```text
Fold 1 : entraîner sur 2+3, produire des scores sur 1
Fold 2 : entraîner sur 1+3, produire des scores sur 2
Fold 3 : entraîner sur 1+2, produire des scores sur 3
```

Chaque observation reçoit ainsi un score produit par un modèle qui ne l'a pas utilisée
pour son propre entraînement. Ces scores `out-of-fold` servent à apprendre la
calibration.

Dans le projet, `CalibratedClassifierCV(..., cv=3)` gère cette logique à l'intérieur de
l'ensemble d'entraînement. L'ensemble de test reste séparé.

### 17.4 Ce que l'on espère obtenir

Une bonne calibration post-hoc doit idéalement :

- améliorer log-loss, Brier et métriques de calibration ;
- conserver autant que possible la discrimination ;
- généraliser sur des observations nouvelles.

Mais aucune méthode ne garantit d'améliorer toutes les métriques dans tous les jeux de
données.

## 18. Platt scaling

### 18.1 Idée intuitive

Platt scaling apprend une courbe en forme de S, appelée **sigmoïde**.

Il part du score brut `s` et apprend deux paramètres `a` et `b` :

```text
p_corrigée = sigmoid(a * s + b)
```

avec :

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

La sigmoïde transforme n'importe quel nombre réel en une valeur comprise entre 0 et 1.

### 18.2 Rôle des paramètres

`a` contrôle principalement la pente :

- grande valeur absolue : transition rapide entre faibles et fortes probabilités ;
- petite valeur absolue : transition plus douce.

`b` déplace la courbe vers la gauche ou la droite et ajuste son niveau général.

Ces deux paramètres sont choisis pour minimiser une log-loss sur les données de
calibration.

### 18.3 Exemple simplifié

Supposons que le modèle brut produise :

```text
0,20 ; 0,40 ; 0,60 ; 0,80
```

Mais les fréquences observées correspondantes soient environ :

```text
0,05 ; 0,15 ; 0,35 ; 0,65
```

Le modèle brut est globalement trop confiant. Platt scaling peut apprendre une sigmoïde
qui abaisse ces valeurs tout en conservant leur ordre.

### 18.4 Pourquoi le classement reste généralement identique

Une sigmoïde correctement orientée est monotone : lorsque le score brut augmente, la
probabilité corrigée augmente aussi.

```text
avant : A > B > C
après : f(A) > f(B) > f(C)
```

L'AUC-ROC change donc peu ou pas lorsque la transformation est appliquée à un modèle
fixe. Dans `CalibratedClassifierCV`, de petites différences peuvent toutefois apparaître
parce que la procédure combine plusieurs modèles entraînés sur différents folds.

### 18.5 Forces et limites

Avantages :

- simple ;
- peu de paramètres ;
- relativement stable avec peu de données.

Limites :

- la forme en S peut être trop rigide ;
- elle ne peut pas suivre une relation irrégulière entre score et fréquence réelle ;
- une bonne amélioration globale peut masquer une détérioration dans une zone
  opérationnelle particulière.

### 18.6 Résultat dans notre projet

Platt scaling réduit l'ECE uniforme :

```text
baseline : 0,00030
Platt    : 0,00016
```

Si l'on regardait uniquement cette métrique, on conclurait que Platt améliore la
calibration.

Mais l'erreur du top 1 % augmente :

```text
baseline : 0,00360
Platt    : 0,01609
```

La log-loss et le Brier score deviennent également légèrement moins bons. La sigmoïde
améliore donc l'accord moyen selon un découpage, mais ajuste mal la queue des scores les
plus élevés.

## 19. Régression isotone

### 19.1 Idée intuitive

La régression isotone apprend une fonction monotone sans imposer une forme en S.

Elle doit respecter :

```text
si score A <= score B,
alors probabilité corrigée A <= probabilité corrigée B
```

Mais entre ces contraintes, elle peut apprendre une fonction en escalier beaucoup plus
flexible.

### 19.2 Exemple

Supposons les scores triés et résultats suivants :

| Score | Résultat réel |
|---:|---:|
| 0,10 | 0 |
| 0,20 | 0 |
| 0,30 | 1 |
| 0,40 | 0 |
| 0,50 | 1 |
| 0,60 | 1 |

Les résultats individuels ne sont pas monotones : à `0,30` on observe 1, puis à `0,40`
on observe 0.

La régression isotone regroupe les zones qui violent l'ordre et leur attribue une
moyenne commune. Elle peut produire quelque chose comme :

```text
0,10 -> 0,00
0,20 -> 0,00
0,30 -> 0,50
0,40 -> 0,50
0,50 -> 1,00
0,60 -> 1,00
```

Le résultat ressemble à un escalier monotone.

### 19.3 Intuition de l'algorithme PAV

Un algorithme classique s'appelle `Pool Adjacent Violators` :

1. trier les observations par score ;
2. regarder les fréquences observées ;
3. si deux blocs voisins violent l'ordre croissant, les fusionner ;
4. remplacer leurs valeurs par leur moyenne ;
5. recommencer jusqu'à obtenir une suite monotone.

Il n'est pas nécessaire de savoir le programmer en entretien, mais il faut comprendre
pourquoi la fonction obtenue est plus flexible qu'une sigmoïde.

### 19.4 Forces et limites

Avantage : plus flexible.

Limites :

- risque de surapprentissage avec peu de données ;
- fonction en escalier parfois instable dans les régions rares ;
- elle ne sait pas extrapoler élégamment au-delà des scores observés ;
- elle demande généralement davantage de données de calibration que Platt.

### 19.5 Résultat dans notre projet

La régression isotone obtient ici :

```text
ECE        : 0,00015
ACE        : 0,00019
Brier      : 0,00050
log-loss   : 0,00321
top-1% CE  : 0,00411
```

Elle améliore plusieurs métriques globales par rapport à la baseline. Son top-1% CE est
proche de la baseline, mais moins mauvais que celui de Platt.

Cela ne signifie pas que la régression isotone est toujours supérieure. Cela signifie
que sa flexibilité correspond mieux à la relation score-fréquence observée dans cette
expérience.

## 19.6 Comparaison directe

| Aspect | Platt scaling | Régression isotone |
|---|---|---|
| Forme | Sigmoïde paramétrique | Escalier monotone non paramétrique |
| Nombre de paramètres | Très faible | Dépend des données |
| Besoin en données | Plus faible | Plus élevé |
| Flexibilité | Limitée | Forte |
| Risque de surapprentissage | Plus faible | Plus élevé |
| Extrapolation | Plus naturelle | Plus délicate |

## 19.7 Calibration post-hoc contre correction du prior

Ces deux opérations peuvent sembler similaires parce qu'elles transforment les scores,
mais leur logique diffère.

### Correction du prior

- utilise une formule analytique ;
- suppose que l'on connaît le mécanisme d'undersampling ;
- corrige un changement précis de prévalence ;
- n'apprend pas ses paramètres à partir des labels de calibration.

### Calibration post-hoc

- apprend une relation entre scores et résultats réels ;
- peut corriger plusieurs formes empiriques de mauvaise calibration ;
- nécessite des données de calibration représentatives ;
- ne donne aucune garantie sous un nouveau changement de distribution.

On peut parfois appliquer une correction de prior, puis encore calibrer post-hoc si des
erreurs subsistent. Mais l'évaluation doit rester faite sur un test indépendant.

## 19.8 Pourquoi la calibration peut échouer après déploiement

Une fonction post-hoc est apprise sur une distribution donnée. Si le monde change :

- nouveau type de fraude ;
- changement de population ;
- nouvelles politiques de contrôle ;
- changement du taux de fraude ;
- nouveau comportement des clients ;

la relation entre score et fréquence réelle peut changer. Le modèle calibré hier peut
être mal calibré demain.

La calibration doit donc être surveillée et éventuellement réestimée.

## 19.9 Cas multiclasse et temperature scaling

Pour un réseau multiclasse, une méthode post-hoc fréquente est le **temperature
scaling**.

Le réseau produit des logits `z_1, ..., z_K`. On les divise par une température positive
`T` avant le softmax :

```text
probabilités = softmax(logits / T)
```

- `T > 1` rend les probabilités moins extrêmes ;
- `T < 1` les rend plus confiantes ;
- l'ordre des classes pour une observation ne change pas.

On choisit `T` sur un ensemble de calibration en minimisant la log-loss. Cette méthode
est simple, mais une seule température globale peut cacher des erreurs spécifiques à
certaines classes.

## 19.10 Lecture du code du projet

Dans la cellule de fonctions du notebook :

```python
def platt_scaling(estimator_factory, X_train, y_train, cv=5):
    calibrated = CalibratedClassifierCV(
        estimator_factory(), method="sigmoid", cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated
```

`method="sigmoid"` demande Platt scaling.

```python
def isotonic_scaling(estimator_factory, X_train, y_train, cv=5):
    calibrated = CalibratedClassifierCV(
        estimator_factory(), method="isotonic", cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated
```

`method="isotonic"` demande la régression isotone.

Dans les cellules d'analyse, le projet utilise `cv=3`, puis applique :

```python
p_platt = platt.predict_proba(X_test)[:, 1]
p_isotonic = isotonic.predict_proba(X_test)[:, 1]
```

`predict_proba(... )[:, 1]` sélectionne la probabilité corrigée de la classe positive,
c'est-à-dire la fraude.

## 19.11 Questions d'entretien

### Pourquoi utiliser un jeu de calibration séparé ?

Pour que la transformation apprenne à corriger des scores produits sur des observations
non utilisées par le classifieur, et éviter une estimation trop optimiste.

### Pourquoi Platt est-il plus stable avec peu de données ?

Parce qu'il n'apprend que deux paramètres et impose une forme sigmoïde.

### Pourquoi isotonic peut-elle mieux fonctionner ?

Parce qu'elle peut suivre une relation monotone plus complexe qu'une sigmoïde.

### Pourquoi isotonic peut-elle sur-apprendre ?

Sa grande flexibilité lui permet aussi de suivre le bruit de l'échantillon de
calibration, surtout dans les zones contenant peu d'observations.

### Est-ce que la calibration post-hoc améliore l'AUC ?

Pas nécessairement. Une transformation monotone conserve le classement. Son objectif
est l'échelle probabiliste, pas la discrimination.

### Pourquoi Platt améliore l'ECE mais détériore le top 1 % dans le projet ?

Parce qu'une sigmoïde globale peut mieux ajuster la masse principale des observations
tout en ajustant moins bien la queue rare des scores élevés. Une moyenne globale ne
garantit pas une amélioration locale.

### Pourquoi le doctorat veut-il aller au-delà du post-hoc ?

Le post-hoc corrige les sorties après apprentissage. Une loss différentiable cherche à
faire apprendre directement au réseau une représentation et des probabilités plus
fiables, potentiellement avec des propriétés mieux adaptées au multiclasse. Mais cette
intégration crée aussi de nouveaux compromis d'optimisation.

## 19.12 Résumé

> La calibration post-hoc garde le classifieur déjà entraîné et apprend ensuite une
> transformation de ses scores sur des données séparées. Platt impose une sigmoïde
> simple et stable ; isotonic apprend une fonction monotone plus flexible mais plus
> sensible au surapprentissage. Leur objectif est d'améliorer les probabilités, pas le
> classement.

---

# Partie VI — Calibration pendant l'entraînement

## 20. Pourquoi ajouter une loss de calibration ?

La calibration post-hoc intervient après l'entraînement. Le sujet du doctorat cherche à
intégrer une exigence de calibration directement dans la fonction objectif :

```text
loss totale = loss de classification + lambda * loss de calibration
```

Dans le projet :

```python
loss = bce(logits, yb)
loss = loss + calibration_weight * soft_ece_loss(probs, yb)
```

`lambda`, appelé `calibration_weight`, règle le compromis.

## 21. Pourquoi l'ECE classique n'est pas différentiable

L'entraînement d'un réseau utilise la descente de gradient. Il faut savoir comment une
petite modification des paramètres modifie la loss.

Dans l'ECE classique, une probabilité est affectée brutalement à un bin. Si elle passe
de `0,1999` à `0,2001`, elle peut changer instantanément de bin. Cette opération dure
n'a pas un gradient utile au niveau de la frontière.

## 22. Idée du soft binning

Au lieu de placer une observation dans un seul bin, on lui donne une appartenance
partielle à plusieurs bins.

Si sa probabilité est `0,42`, elle peut recevoir :

```text
bin centré à 0,30 -> petit poids
bin centré à 0,40 -> grand poids
bin centré à 0,50 -> poids moyen
```

Les poids sont calculés avec un noyau gaussien :

```python
weights = exp(-(proba - centre_bin)^2 / (2 * temperature^2))
```

Puis ils sont normalisés pour que leur somme vaille 1.

La transition devient continue : une petite modification de la probabilité produit une
petite modification des poids. La loss devient différentiable.

## 23. Calcul de la soft-ECE du projet

Dans chaque bin souple :

```python
bin_conf = moyenne pondérée des probabilités
bin_acc  = moyenne pondérée des labels
```

Puis :

```python
soft_ECE = somme(poids_du_bin * |bin_acc - bin_conf|)
```

Cette loss est une illustration simplifiée. Elle n'est pas une nouvelle méthode et ne
reproduit pas exactement un article scientifique particulier.

## 24. Le danger de la prédiction constante

Imaginons une population avec 1 % de positifs. Un modèle qui prédit toujours `0,01` est
globalement bien calibré : sa moyenne correspond au taux réel. Pourtant, il ne distingue
absolument aucun cas.

Une loss de calibration trop forte peut pousser le réseau vers cette solution facile.
La BCE encourage la discrimination ; la loss de calibration encourage l'accord entre
confiance et fréquence. Il faut contrôler leur compromis.

---

# Partie VII — Lecture du code

## 25. `calibration_walkthrough.ipynb`

C'est à la fois le cours pratique et le chef d'orchestre. Toutes les fonctions et toutes
les expériences sont regroupées dans ce notebook autonome.

Il réalise dans l'ordre :

1. chargement des données ;
2. Random Forest de référence ;
3. Platt scaling et régression isotone ;
4. expérience d'undersampling et correction du prior ;
5. entraînement des deux MLP ;
6. calcul des métriques ;
7. création des graphiques et du tableau de résultats.

La fonction `full_report` évite de répéter le calcul de toutes les métriques pour chaque
méthode.

## 26. Fonctions de chargement dans le notebook

### `_load_local`

Cherche `data/creditcard.csv`.

### `_load_openml`

Télécharge la copie publique si le CSV local n'existe pas.

### `_make_synthetic`

Crée un jeu artificiel si les vraies données sont indisponibles. Il sert à garantir que
le pipeline peut fonctionner hors ligne. Les résultats synthétiques ne doivent pas être
présentés comme ceux du vrai dataset.

### `load_data`

Sépare `X` et `y`, puis crée train et test avec `stratify=y`.

## 27. Fonctions de calibration dans le notebook

### `make_base_estimator`

Construit la Random Forest avec une seed fixe pour la reproductibilité.

### `discrimination_report`

Calcule AUC-ROC, AUC-PR, Brier et log-loss.

### `undersample`

Conserve tous les positifs et seulement une partie des négatifs. Retourne également
`beta`, nécessaire à la correction.

### `correct_undersampled_probabilities`

Replace les probabilités sur l'échelle de la population réelle.

### `platt_scaling` et `isotonic_scaling`

Enveloppent la Random Forest dans `CalibratedClassifierCV`.

## 28. Fonctions de métriques dans le notebook

### `expected_calibration_error`

Crée des bins de largeur égale puis calcule l'écart pondéré.

### `adaptive_calibration_error`

Trie les probabilités et construit des groupes d'effectif similaire.

### `top_fraction_calibration_error`

Sélectionne exactement les `k` scores les plus élevés avec `argpartition`, puis compare
probabilité moyenne et fréquence réelle.

### `reliability_curve`

Retourne les coordonnées nécessaires au diagramme de fiabilité.

### `plot_reliability`

Trace la diagonale parfaite et les résultats. La taille des marqueurs dépend du nombre
d'observations dans chaque bin.

## 29. Fonctions PyTorch dans le notebook

### `MLP`

Petit réseau avec deux couches cachées et une sortie appelée **logit**.

Un logit est un nombre réel quelconque. La fonction sigmoïde le transforme en
probabilité entre 0 et 1 :

```python
probs = torch.sigmoid(logits)
```

### `soft_ece_loss`

Construit les bins souples et mesure l'écart calibration-confiance.

### `train_mlp`

- fixe la seed ;
- choisit CPU ou GPU ;
- convertit les tableaux en tenseurs ;
- mélange les observations à chaque époque ;
- calcule la BCE et éventuellement la soft-ECE ;
- effectue `backward()` puis `optimizer.step()` ;
- renvoie les probabilités de validation/test.

## 30. Pourquoi il n'y a plus de dossier `src`

Le dépôt a été simplifié pour ne conserver qu'un seul support de code. Les anciennes
fonctions Python séparées ont été intégrées directement dans le notebook. Cela facilite
la lecture séquentielle, au prix d'une structure moins modulaire qu'un package Python.

## 31. Pourquoi `.gitignore` existe

Il évite de versionner :

- les données brutes volumineuses ;
- l'environnement Python local ;
- les fichiers temporaires générés par Python ou Jupyter.

Il ne supprime aucun fichier du disque.

---

# Partie VIII — Comprendre les résultats

## 32. Random Forest initiale

Elle discrimine bien et sa calibration globale est déjà relativement bonne. C'est une
leçon importante : on ne doit pas choisir après coup un récit qui arrange le projet.

## 33. Méthodes post-hoc

La régression isotone obtient ici de bons résultats globaux. Platt scaling améliore
l'ECE uniforme mais détériore l'erreur dans le top 1 %. Une seule métrique aurait donné
une conclusion incomplète.

## 34. Undersampling

Les probabilités brutes sont fortement faussées. La correction analytique améliore
massivement log-loss, Brier et calibration. Elle ne récupère cependant pas entièrement
la qualité du modèle initial et ne change pas son classement.

## 35. Soft-ECE

Après application de la même correction de prior aux deux MLP :

- le top-1% calibration gap s'améliore fortement ;
- log-loss et Brier s'améliorent très légèrement ;
- ECE et ACE globales se détériorent légèrement ;
- AUC-ROC baisse un peu ;
- AUC-PR augmente légèrement.

Conclusion : la loss déplace le compromis entre plusieurs propriétés. Elle ne gagne pas
partout.

---

# Partie IX — Du binaire au multiclasse

## 36. Le cas multiclasse

Dans le binaire, on prédit une probabilité `p` pour la classe positive et `1-p` pour la
classe négative.

Dans un problème à trois classes :

```text
classe A : 0,60
classe B : 0,30
classe C : 0,10
somme    : 1,00
```

La calibration devient plus complexe.

## 37. Plusieurs notions de calibration

### Calibration de confiance

On regarde seulement la probabilité maximale et si la classe choisie est correcte.

### Calibration classwise

Pour chaque classe séparément, une probabilité de 30 % doit correspondre à environ
30 % d'appartenance réelle à cette classe.

### Calibration marginale ou jointe

On étudie des propriétés plus fortes du vecteur complet de probabilités. Plus la notion
est forte, plus elle est difficile à estimer avec un nombre limité d'observations.

Un modèle peut sembler bien calibré selon la confiance maximale tout en étant mauvais
pour une classe rare particulière.

## 38. Pourquoi une loss multiclasse est difficile

- Les probabilités sont liées par la contrainte de somme à 1.
- Le nombre de régions possibles dans le simplex augmente avec les classes.
- Certaines classes peuvent être rares.
- Une métrique agrégée peut cacher les erreurs classwise.
- Le soft binning devient multidimensionnel ou doit être défini selon une notion précise
  de calibration.
- Optimiser directement une métrique empirique peut créer des solutions dégénérées.

C'est ici que le projet rejoint le sujet du doctorat.

---

# Partie X — Préparation orale

## 39. Présentation du projet en 30 secondes

> J'ai construit une petite étude reproductible sur la calibration en détection de
> fraude déséquilibrée. J'ai comparé calibration post-hoc, correction analytique après
> undersampling et une loss soft-ECE intégrée à un réseau. L'expérience m'a surtout
> montré que discrimination et calibration sont différentes, que l'undersampling
> change l'échelle probabiliste, et qu'une loss de calibration peut améliorer une zone
> opérationnelle tout en détériorant d'autres métriques. Le projet reste binaire et
> exploratoire ; son extension multiclasse est la question qui m'intéresse.

## 40. Présentation en deux minutes

Structure conseillée :

1. **Problème** : score de fraude différent d'une probabilité fiable.
2. **Données** : 284 807 transactions, environ 0,17 % de fraudes.
3. **Méthodes** : Random Forest, Platt, isotonic, undersampling/correction, deux MLP.
4. **Évaluation** : AUC-ROC/PR, log-loss, Brier, ECE, ACE, top 1 %.
5. **Résultat** : correction du prior essentielle ; soft-ECE donne un compromis.
6. **Limites** : une seed, binaire, hyperparamètre exploratoire.
7. **Suite** : répétitions, validation, métriques classwise, multiclasse.

## 41. Questions auxquelles tu dois répondre sans notes

1. Quelle différence entre score, probabilité et décision ?
2. Un modèle peut-il avoir une excellente AUC et être mal calibré ? Pourquoi ?
3. Comment lire un diagramme de fiabilité ?
4. Pourquoi l'ECE dépend-elle des bins ?
5. Pourquoi l'accuracy est-elle trompeuse ici ?
6. Pourquoi l'undersampling modifie-t-il les probabilités ?
7. Pourquoi la correction ne modifie-t-elle pas l'AUC ?
8. Quelle différence entre Platt et isotonic ?
9. Pourquoi l'ECE dure n'est-elle pas une bonne loss de gradient ?
10. Comment fonctionne le soft binning ?
11. Pourquoi une prédiction constante peut-elle sembler calibrée ?
12. Pourquoi appliquer la même correction aux deux MLP ?
13. Quel est le résultat principal du projet ?
14. Quelles sont ses limites ?
15. Pourquoi le multiclasse est-il plus difficile ?

## 42. Petit exercice mental

Pour vérifier que tu as compris, réponds avant de regarder la solution :

Un modèle attribue `0,80` à 100 observations. Parmi elles, 50 sont positives.

1. Est-il bien calibré dans ce groupe ?
2. Est-il surconfiant ou sous-confiant ?
3. Quel est l'écart de calibration ?

Solution :

1. Non, car la fréquence réelle vaut `50/100 = 0,50`.
2. Il est surconfiant : il annonce 80 % alors que 50 % sont positives.
3. L'écart vaut `|0,50 - 0,80| = 0,30`.

---

# Résumé final à retenir

1. Un score bien classé n'est pas forcément une probabilité fiable.
2. Calibration signifie accord entre probabilités annoncées et fréquences observées.
3. AUC mesure surtout le classement ; Brier, log-loss, ECE et ACE regardent d'autres
   aspects.
4. L'undersampling change le prior et fausse les probabilités brutes.
5. Une correction analytique peut remettre ces probabilités sur l'échelle réelle sous
   certaines hypothèses.
6. Platt et isotonic calibrent après entraînement.
7. Une loss différentiable cherche à intégrer la calibration pendant l'entraînement.
8. Optimiser la calibration seule peut détruire la discrimination.
9. Les métriques peuvent être en désaccord ; il faut expliquer pourquoi.
10. Le multiclasse introduit plusieurs notions de calibration et constitue un problème
    de recherche plus difficile.
