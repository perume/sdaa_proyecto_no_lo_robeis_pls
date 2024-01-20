import cv2
import mediapipe as mp
import numpy as np

fingers = {0:"Pulgar", 1:"Indice", 2:"Corazon",3:"Anular",4:"Menique"}


mp_hands = mp.solutions.hands.Hands()
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

vid = cv2.VideoCapture(0)

def get_fingercode(landmarks : list):
    """Takes list of hand_landmarks and returns tuple showing which fingers are extended (joints roughly align) and which dont
    Tuple order is (p,i,c,a,m), 1 if extended, 0 otherwise
    Landmarks is a results.multi_hand_landmarks list"""
    fingercode = []
    if landmarks is not None:
        for detected_hand in landmarks:
            for finger in range(5):
                art1 = np.array([detected_hand.landmark[1+finger*4].x, detected_hand.landmark[1+finger*4].y,detected_hand.landmark[1+finger*4].z], np.float16)
                art2 = np.array([detected_hand.landmark[2+finger*4].x, detected_hand.landmark[2+finger*4].y,detected_hand.landmark[2+finger*4].z], np.float16)
                art3 = np.array([detected_hand.landmark[3+finger*4].x, detected_hand.landmark[3+finger*4].y,detected_hand.landmark[3+finger*4].z], np.float16)
                art4 = np.array([detected_hand.landmark[4+finger*4].x, detected_hand.landmark[4+finger*4].y,detected_hand.landmark[4+finger*4].z], np.float16)
                print("Points")
                print(art1)
                print(art2)
                print(art3)
                print(art4)
                print("Vectors")
                print(art2-art1)
                print(art3-art2)
                print(art4-art3)
                vec1 = (art2 - art1) / np.linalg.norm(art2 - art1)
                vec2 = (art3 - art2) / np.linalg.norm(art3 - art2)
                vec3 = (art4 - art3) / np.linalg.norm(art4 - art3)
                print("Normed vectors")
                print(vec1)
                print(vec2)
                print(vec3)
                ang1 = np.rad2deg(np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)))
                ang2 = np.rad2deg(np.arccos(np.clip(np.dot(vec2, vec3), -1.0, 1.0)))
                print("Finger "+fingers[finger]+" has angles: "+str(ang1)+" "+str(ang2))
                

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    
    frame.flags.writeable = False
    results = mp_hands.process(frame)
    frame.flags.writeable = True
    
    if results.multi_hand_landmarks:
        get_fingercode(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hand_connections,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands',cv2.flip(frame, 1))
    get_fingercode(results.multi_hand_landmarks)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        #import pdb; pdb.set_trace()
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 