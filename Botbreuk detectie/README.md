Het doel van dit project is het identificeren van scheuren in boten met behulp van Convolutional Neural Networks (CNNs). 
Voor de implementatie van het model zijn TensorFlow en Scikit-Image de belangrijkste libraries die gebruikt zijn. Verder uitleg van het project wordt in het bestand Computer Vision Challenge uitgewerkt.

De gebruikte DataSet is van Roboflow 
1. bone fracture detection > v4
https://universe.roboflow.com/veda/bone-fracture-detection-daoon
Instructies:
1. Download de laatste dataset uit deze link ->https://universe.roboflow.com/veda/bone-fracture-detection-daoon
   Zorg ervoor dat de map 'BoneFractureYolo8' naast bone ```fracture detection.v4-v4.yolov8``` staat.
3. Run ```preprocessFromKaggle.py``` om de gegenereerde train, validation en test subset te genereren
4. Run ```loadPreprocessed.py``` om het model te trainen. Hier kunnen ook hyper parameters worden aangepast.
5. Run ```test_model``` om de confusion matrix en de sliding window methode te gebruiken voor evaluatie van het model
