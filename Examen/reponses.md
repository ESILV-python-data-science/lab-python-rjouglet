# Classification des pages de livres d’heures

## Question : Prenez soin de bien définir les différents ensembles (train/validation/test). L’objectif est de pouvoir estimer la performance du classifieur (taux de classification correcte) sur des nouveaux manuscrits. Donnez les résultats de toutes vos expériences, même les résutats négatifs. Faites une analyse votre résultat final, telle qu’elle pourrait être transmise à votre client.



Après avoir lu le fichier .csv, on utilise la fonction Extract Features afin de faciliter et raccourcir le temps d'éxécution du programme. 
On applique ensuite un SVC sur un echantillon de valeurs ce qui nous permet d'obtenir des résultats plutôt corrects.

calendrier       0.71      0.62      0.67        16
miniature       1.00       0.50      0.67        2

On obtient alors une précision de 0.91 pour un k=1



