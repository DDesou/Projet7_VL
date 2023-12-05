# OC_Project7
## Projet OpenClassroom parcours Data Scientist
### Description du dossier API
Le choix a été fait de construire une API avec **FastAPI** déployée à travers une Web App sous la plateforme cloud Azure (Microsoft).
En plus du dossier (fichier Yaml) de workflows, décrivant les différentes tâches à effectuer, plusieurs fichiers sont indispensables :
- **le script python de l'API noté 'main.py'** détaillé plus bas
- **le script python du test unitaire (Pytest) noté 'test_main.py'** : ce dernier test lors du déploiement continue que l'API appelle bien le modèle pour effectuer des prédictions de classement (un individu en-dessous et un au-dessus sont testés)
- **les deux datasets réduits (dans un dossier 'ressources')** contenant respectivement les informations des clients références et celles des nouveaux clients. Ces datasets sont amenés à évoluer avec le temps et ne seront modifiés qu'ici.
- **le fichier 'model.joblib' (dans 'ressources') dans lequel le modèle entraîné est enregistré, ainsi que le pipeline de transformation des données**. Si le modèle change ou le pipeline changent, le fichier sera changé.
- **le fichier requirements.txt contenant les librairies nécessaires** à contruire l'environnement dans lequel l'API peut fonctionner (à modifier en cas d'utilisation de nouveaux modèles de références par exemple).
- **/!\ A noter qu'un script de lancement de l'API sous Azure est nécessaire (Configuration/General Settings/Startup command)** : 
> apt update
> apt-get install -y libgomp1
> gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app

A noter que l'API doit être installée sous une **Web App Azure Linux en Python 3.11**.
L'URL de l'application est la suivante : **https://basicwebappvl.azurewebsites.net**. Bien entendu, pour qu'elle fonctionne, elle nécessite d'être activée!

Concernant le script de l'API, **différentes "routes" sont mises en place** pour interagir avec l'utilisateur de l'application (interface du conseiller clientèle).
**Différentes méthodes GET** pour obtenir :
- la liste des ids des nouveaux clients pour vérifier que le client est bien enregistré
- la liste des features qui sont utilisées pour la modélisation
- envoyer le numéro du client sélectionné et recevoir ses informations (format json, données brutes)
- envoyer le numéro du client sélectionné et recevoir ses informations (format json, données transformées)
- envoyer le json d'un client (données transformées) et recevoir sa prédiction de classement
- envoyer le numéro d'un client et recevoir son explicabilité locale (Shap values)
- envoyer le nom d'une feature et recevoir les valueurs de cette features pour les individus classés 0 et ceux classés 1


### Tests et workflows
La plateforme d'hébergement choisie est **Azure Web App**.  
Afin de mettre en place un **processus d'intégration/amélioration continues**, le code est hébergé sur des **repo Git distants** et le déploiement réalisé par les **actions Github** communiquant avec l'hébergeur. De cette manière, des modifications peuvent être réalisées puis contrôlées d'abord dans un **environnement virtuel local** défini, puis éventuellement déployées dans une **nouvelle branche** avant d'être envoyées à la branche principale.  
Il a été décidé de séparer complètement le déploiement de l'API de celui de l'application et des projets Github distincts ont été créés :
- pour l'API : https://github.com/DDesou/Projet7_VL
- pour l'interface utilisateur : https://github.com/DDesou/Projet7_Streamlit

L'API est déployée (ou modifiée) après que les tests unitaires aient été validés (**tests Pytest** intégrés dans le déploiement). Les scripts des test effectués sont contenus dans le fichier test_main.py.  
L'application Streamlit est déployée (ou modifiée) après s'être assuré que toutes les modifications ont été d'abord enregistrées, afin d'éviter les bugs éventuels au moment du changement de version.
