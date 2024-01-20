import cv2
import mediapipe as mp
import numpy as np

fingers_dict = {0:"Pulgar", 1:"Indice", 2:"Corazon",3:"Anular",4:"Menique"}
debug = 0

mp_hands = mp.solutions.hands.Hands()
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

vid = cv2.VideoCapture(0)

def get_fingercode(landmarks : list, threshold_thumb = 25, threshold = 50):
    """Takes list of hand_landmarks and returns tuple showing which fingers are extended (joints roughly align) in the first detected hand and which dont
    Tuple order is (p,i,c,a,m), 1 if extended, 0 otherwise
    Landmarks is a results.multi_hand_landmarks list"""
    fingercode = [0,0,0,0,0]
    if landmarks is not None:
        for detected_hand in landmarks:
            for finger in range(5):
                joints = [None,None,None,None]
                for joint in range(4):
                    joints[joint] = np.array([detected_hand.landmark[joint+1+finger*4].x, detected_hand.landmark[joint+1+finger*4].y,detected_hand.landmark[joint+1+finger*4].z], np.float16)
                if debug:
                    print("Points")
                    for joint in joints:
                        print(joint)
                    print("Vectors")
                    for i in range(3):
                        print(joints[i+1]-joints[i])
                vectors = [None, None, None]
                for i in range(3):
                    vectors[i] = (joints[i+1] - joints[i]) / np.linalg.norm(joints[i+1] - joints[i])
                if debug:
                    print("Normed vectors")
                    for vector in vectors:
                        print(vector)
                angles = [None, None]
                angles[0] = np.rad2deg(np.arccos(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0)))
                angles[1] = np.rad2deg(np.arccos(np.clip(np.dot(vectors[1], vectors[2]), -1.0, 1.0)))
                if debug:
                    print("Finger "+fingers_dict[finger]+" has angles: "+str(angles))
                if finger == 0:
                    min_angle = threshold_thumb
                else:
                    min_angle = threshold
                if ((angles[0]+angles[1])/2 < min_angle):
                    fingercode[finger] = 1
        return fingercode
    return None

def get_contact(landmarks : list, threshold = 0.1):
    """Takes list of hand_landmarks and returns tuple of touching fingertips
        We assume the thumb is always going to be one of the fingers involved
        If your anatomy lets you touch multiple fingertips comfortably on one hand go to a chiropractor ig
        Issue: Threshold is not normalized
    """
    if landmarks is not None:
        for detected_hand in landmarks:
            fingertips = [None,None,None,None,None]
            for finger_index in range(5):
                fingertips[finger_index] = np.array([detected_hand.landmark[4*finger_index+4].x,detected_hand.landmark[4*finger_index+4].y,detected_hand.landmark[4*finger_index+4].z])
            if debug:
                print("Fingertips")
                print(fingertips)
            distances = [None,None,None,None]
            for i in range(4):
                distances[i] = np.linalg.norm(fingertips[i+1]-fingertips[0])
            if debug:
                print("Distances")
                print(distances)
                print("Closest: "+fingers_dict[np.argmin(distances)+1])
            if np.amin(distances) < threshold:
                return np.argmin(distances)+1
            else:
                return None

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    
    frame.flags.writeable = False
    results = mp_hands.process(frame)
    frame.flags.writeable = True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hand_connections,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands',cv2.flip(frame, 1))
    fingercode = get_fingercode(results.multi_hand_landmarks)  
    contacts = get_contact(results.multi_hand_landmarks)
    print(str(fingercode) + " - " + str(contacts))  
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