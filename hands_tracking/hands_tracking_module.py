import cv2                                                                                                                  # Utilisé pour récupérer le live de la caméra
import mediapipe as mp                                                                                                      # utilisé pour détecter les mains


class HandsDetector:

    def __init__(self,
                 static_image_mode=False,                                                                                   # Tant que le taux de confiance est bon : pas de recalcul systématique
                 max_num_hands=2,                                                                                           # Nombre max de mains à détecter dans une image
                 model_complexity=1,                                                                                        # ???
                 min_detection_confidence=0.5,                                                                              # Taux de confiance minimum pour la détetcion : 50%
                 min_tracking_confidence=0.5):                                                                              # Taux de confiance minimum pour le tracking : 50%

        self.static_image_mode = static_image_mode                                                                          # Paramètres pour la classe Hands de mediapipe
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.hand_detection_results = None                                                                                  # Pour détecter les mains
        self.mediapipe_Hands = mp.solutions.hands
        self.hands = self.mediapipe_Hands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                                self.min_detection_confidence, self.min_tracking_confidence)

        self.mediapipe_Draw = mp.solutions.drawing_utils                                                                    # Dessiner les points de repères (landmarks) détectés sur la main

    def find_hands(self, camera_image, draw_landmarks=True):
        camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)                                                    # Convertir l'image de la caméra en objet RGB (nécessaire pour hands.process())
        self.hand_detection_results = self.hands.process(camera_image_rgb)                                                  # Détecter une main et obtenir les valeurs
        if self.hand_detection_results.multi_hand_landmarks:                                                                # Si on a des valeurs (différent de None) : alors on a bien détecté une main
            for hand in self.hand_detection_results.multi_hand_landmarks:                                                   # Boucle pour chaque main détectée
                if draw_landmarks:
                    self.mediapipe_Draw.draw_landmarks(camera_image, hand, self.mediapipe_Hands.HAND_CONNECTIONS)           # Déssiner les points de repère (landmarks) de la main avec les connexions entre points
        return camera_image

    def find_position(self, camera_image, hand_number=0, draw_positions=True):
        landmarks_list = []                                                                                                 # Liste des points de repères (landmarks) à retourner
        if self.hand_detection_results.multi_hand_landmarks:                                                                # Pour une main en particulier
            hand = self.hand_detection_results.multi_hand_landmarks[hand_number]                                            # Récupérer les informations de la main (points de repères + coordonnées)
            for id, landmarks in enumerate(hand.landmark):                                                                  # id correspond au 21 points de repères
                longueur, largeur, cannaux = camera_image.shape                                                             # Calculer le ratio / l'échelle de notre image
                position_x, position_y = int(landmarks.x * largeur), int(landmarks.y * longueur)                            # Trouver une position dans l'écran
                landmarks_list.append([id, position_x, position_y])                                                         # Ajouter ces coordonées dans la liste
                if draw_positions:                                                                                          # Marquer cette position à l'écran
                    cv2.circle(camera_image, (position_x, position_y), 25, (255, 0, 255), cv2.FILLED)                       # Dessiner un cercle en définissant sa position, son rayon et sa couleur (cv2.FILLED = forme pleinne)
        return landmarks_list
