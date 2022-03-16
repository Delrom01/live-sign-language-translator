import cv2                                                                                                                  # Utilisé pour récupérer le live de la caméra
import mediapipe as mp                                                                                                      # utilisé pour détecter les visages


class FacesDetector:

    def __init__(self,
                 static_image_mode=False,                                                                                   # Tant que le taux de confiance est bon : pas de recalcul systématique
                 max_num_faces=2,                                                                                           # Nombre max de visages à détecter dans une image
                 refine_landmarks=False,                                                                                    # ???
                 min_detection_confidence=0.5,                                                                              # Taux de confiance minimum pour la détetcion : 50%
                 min_tracking_confidence=0.5):                                                                              # Taux de confiance minimum pour le tracking : 50%

        self.static_image_mode = static_image_mode                                                                          # Paramètres pour la classe FaceMesh de mediapipe
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.face_detection_results = None                                                                                  # Pour détecter les visages
        self.mediapipe_Faces = mp.solutions.face_mesh
        self.faces = self.mediapipe_Faces.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks,
                                                   self.min_detection_confidence, self.min_tracking_confidence)

        self.mediapipe_Draw = mp.solutions.drawing_utils                                                                    # Dessiner les points de repères (landmarks) détectés sur le visage
        self.mediapipe_Draw_Spec = self.mediapipe_Draw.DrawingSpec(thickness=1, circle_radius=1, color=[127, 255, 0])       # Changer l'épaisseur, le rayon et la couleur des points de repères (landmarks) sur le visage

    def find_faces(self, camera_image, draw_landmarks=True):
        camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)                                                    # Convertir l'image de la caméra en objet RGB (nécessaire pour faces.process())
        self.face_detection_results = self.faces.process(camera_image_rgb)                                                  # Détecter un visage et obtenir les valeurs
        if self.face_detection_results.multi_face_landmarks:                                                                # Si on a des valeurs (différent de None) : alors on a bien détecté un visage
            for face in self.face_detection_results.multi_face_landmarks:                                                   # Boucle pour chaque visage détecté
                if draw_landmarks:
                    self.mediapipe_Draw.draw_landmarks(camera_image, face, self.mediapipe_Faces.FACEMESH_CONTOURS,          # Déssiner les points de repère (landmarks) du visage avec les connexions entre points
                                                       self.mediapipe_Draw_Spec, self.mediapipe_Draw_Spec)
        return camera_image

    def find_position(self, camera_image, face_number=0, draw_positions=True, landmark_number=None):
        landmarks_list = []                                                                                                 # Liste des points de repères (landmarks) à retourner
        if self.face_detection_results.multi_face_landmarks:                                                                # Pour un visage en particulier
            face = self.face_detection_results.multi_face_landmarks[face_number]                                            # Récupérer les informations du visage (points de repères + coordonnées)
            for id, landmarks in enumerate(face.landmark):                                                                  # id correspond au 468 points de repères
                longueur, largeur, cannaux = camera_image.shape                                                             # Calculer le ratio / l'échelle de notre image
                position_x, position_y = int(landmarks.x * largeur), int(landmarks.y * longueur)                            # Trouver une position dans l'écran
                landmarks_list.append([id, position_x, position_y])                                                         # Ajouter ces coordonées dans la liste
                if draw_positions:                                                                                          # Si on souhaite marquer des positions
                    if landmark_number:
                        if id == landmark_number:                                                                           # Si une position particulière a été indiquée, marquer cette position à l'écran
                            cv2.circle(camera_image, (position_x, position_y), 25, (255, 0, 255), cv2.FILLED)               # Dessiner un cercle en définissant sa position, son rayon et sa couleur (cv2.FILLED = forme pleinne)
                    else:
                        cv2.circle(camera_image, (position_x, position_y), 25, (255, 0, 255), cv2.FILLED)                   # Sinon, marquer toutes les positions
        return landmarks_list
