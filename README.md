# TER34_Mercantour_park_animal_detection
Système de Deep Learning pour la reconnaissance d’animaux dans des images et des videos du parc du Mercantour

Ce projet est basé sur des réseaux de neurones convolutifs appelés en anglias Convolutionnal Neural
Networks et abrégé CNN. Il s’agit d’un type de réseaux de neurones artificiels acycliques dont le mo-
tif de connexion est inspiré du cortex visuel des animaux. Les neurones de cette région du cerveau
sont arrangés de sorte qu’ils correspondent à des régions qui se chevauchent lors du pavage du champ
visuel. Leur fonctionnement est inspiré par les processus biologiques, ils consistent en un empilage
multicouche de perceptrons, dont le but est de prétraiter de petites quantités d’informations. Les
réseaux neuronaux convolutifs ont de larges applications dans la reconnaissance d’image et vidéo,
les systèmes de recommandation et le traitement du langage naturel.

Detectron2 est une librairie développée par Facebook pour faire de la détection d’objets, de per-
sonnes et d’animaux et de la segmentation. Cette librairie succède à Detectron et MaskRCNN.
MaskRCNN est un algorithme de segmentation avec détection d’objets qui est basé sur l’algorithme
de FastRCNN. Les deux étapes de FastRCNN sont
1. la première étape contient en deux réseaux de neurones un backbone et un regional proposal
network. Ces neurones parcourent une fois par image pour pour donner une liste de proposi-
tions de régions. Les propositions de régions sont les regions qui contiennent les objets.
2. la deuxième étape contient en un réseau de neurones qui prédit les boites et les classes (les
noms des espèces d’animaux ici) pour chaque région détectée dans la première étape. Chaque
région proposée peut être de taille différente alors que des couches complètement connectées
dans les réseaux demande une taille fixe pour faire les prédictions. Ce problème de taille des
régions proposées est réslu en utilisant des fonctions comme RolPool et RolAlign
