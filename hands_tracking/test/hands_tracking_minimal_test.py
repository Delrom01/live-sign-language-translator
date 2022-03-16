# Code minimum pour le tracking

import cv2  # Utilisé pour avoir la caméra en live
import mediapipe as mp  # utilisé pour détecter les mains
import time

# Gérer les FPS :
previousTime = 0
courantTime = 0

# Détecter une main :
mpHands = mp.solutions.hands
hands = mpHands.Hands()     # paramètres par défauts :
# static_image_mode=false (si le taux de confiance est suffisant, il suit la main sans recalculer à chaque fois)
# max_num_hands=2 : nombre max de mains qu'on veut détecter en une image
# min_detection_confidence=0.5 : taux de confiance minimum dans la reconnaissance = 50%
# min_tracking_confidence=0.5 : taux de confiance minimum dans le suivit = 50%
# Si on déscend en dessous de ces taux, on relance la détection

mpDraw = mp.solutions.drawing_utils     # Déssiner les points de repères (landmarks) détectés sur la main

# Obtenir le live de la caméra :
camera = cv2.VideoCapture(0)

while True:
    success_camera_access, camera_image = camera.read()

    # Retourner l'image mirroir
    camera_image = cv2.flip(camera_image, 1)

    ##### Pour les mains
    # Convertir l'image en objet RGB pour hands
    camera_image_RGB = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
    results = hands.process(camera_image_RGB)

    # Détecter une main et obtenir les valeurs
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # Boucle "pour chaque main"
            # Déssiner les points de repère d'une main avec les connexion entre points
            mpDraw.draw_landmarks(camera_image, hand, mpHands.HAND_CONNECTIONS)

            # Récupérer les informations de chaque mains (points de repères + coordonnées)
            for id, landmarks in enumerate(hand.landmark):
                # id correspond au 21 points de repères
                # Calculer le ratio / l'échelle de notre image :
                longueur, largeur, cannaux = camera_image.shape

                # Trouver une position dans l'écran :
                position_x, position_y = int(landmarks.x*largeur), int(landmarks.y*longueur)

                # Marquer cette position à l'écran :
                if id == 8:     # Pour le repère 8
                    # Dessiner un cercle en définissant sa position, son rayon et sa couleur
                    # cv2.FILLED = forme pleinne
                    cv2.circle(camera_image, (position_x, position_y), 25, (255, 0, 255), cv2.FILLED)

    # Gérer les FPS (fréquences d'images) :
    courantTime = time.time()
    FPS = 1/(courantTime-previousTime)
    previousTime = courantTime

    # Arrondir le temps et afficher dans une retour caméra
    # On donne la position, la police, la taille et la couleur et l'épaisseur du texte
    cv2.putText(camera_image, str(int(FPS)), (15, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Afficher à l'écran
    cv2.imshow("Retour caméra en direct", camera_image)
    cv2.waitKey(1)  # Durée de vie de la fenêtre entre sa création et sa destruction
