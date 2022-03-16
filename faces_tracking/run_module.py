import cv2                                                                                                                  # Utilisé pour récupérer le live de la caméra
import time                                                                                                                 # Utilisé pour obtenir la composante temps

from faces_tracking_module import FacesDetector


def main():
    camera = cv2.VideoCapture(0)                                                                                            # Récupérer le live de la caméra
    previous_time = 0                                                                                                       # Gérer les FPS (fréquence d'images par seconde)
    detector = FacesDetector()
    while True:
        success_camera_access, camera_image = camera.read()                                                                 # Récupérer l'image de la caméra
        camera_image = detector.find_faces(camera_image, draw_landmarks=True)                                               # Détecter le visage et dessiner la représentation
        landmarks_list = detector.find_position(camera_image, draw_positions=False)                                         # Récupérer les positions des différents repères d'un visage et les afficher
        # if len(landmarks_list) != 0:
            # print(landmarks_list[8])                                                                                      # Afficher les coordonnées du point de repère 8
        camera_image = cv2.flip(camera_image, 1)                                                                            # Retourner l'image (car effet mirroir)
        courant_time = time.time()                                                                                          # Gérer les FPS
        fps = 1 / (courant_time - previous_time)
        previous_time = courant_time
        cv2.putText(camera_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)                   # Arrondir le temps et afficher dans le retour caméra (on donne la position, la police, la taille et la couleur et l'épaisseur du texte)
        cv2.imshow("Retour caméra en direct", camera_image)                                                                 # Afficher le retour caméra avec les FPS
        cv2.waitKey(1)                                                                                                      # Durée de vie de la fenêtre entre sa création et sa destruction


if __name__ == "__main__":
    main()