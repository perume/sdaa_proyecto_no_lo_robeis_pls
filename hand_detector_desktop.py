import cv2
import mediapipe as mp
import numpy as np
import letter_descriptor as ld
pi = True
if pi:
    #Full functionality is only enabled in the pi
    from picamera2 import Picamera2
    from sense_hat import SenseHat
    from collections import deque
    from time import clock_gettime as gt, CLOCK_REALTIME as rt_clock

debug = 1 #Outputs debug info the higher the value up to 3
show_video = True #Enables opencv video window
if debug>2:
    import timeit

#Instance hand-detection-related objects and variables
mp_hands = mp.solutions.hands.Hands()
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
gc = ld.GestureClassifier()

class LEDFSM():
    """Implements state machine that updates the LED matrix and manages/shows the messages input via gestures"""
    state_dict = {0 : "Idle", 1 : "Recording", 2:"Recorded", 3:"Displaying"}
    color_dict = {0:(255,0,0), 1:(0,0,255), 2:(0,255,0), 3:(255,255,0), 4:(0,0,0)}
    color_dict_halved = {0:(180,0,70), 1:(70,0,180), 2:(20,120,20), 3:(120,120,40), 4:(0,0,0)}
    saved_detections = 10
    time_until_selected = 2.5
    time_until_confirmed = 1
    
    def __init__(self) -> None:
        self.sense = SenseHat()
        self.state = 0
        self.detections = deque(maxlen=self.saved_detections)
        self.detection_time = None
        self.result_string = ""
        self.no_detection = False
    
    def set_led(self, id, halved = False):
        """Sets the LED value to:
        0 - Red
        1 - Blue
        3 - Green
        4 - Yellow"""
        if halved:
            self.sense.set_pixel(0, 0, self.color_dict_halved[id])
        else:
            self.sense.set_pixel(0, 0, self.color_dict[id])
        
    def update_queue(self, value):
        """Adds value to queue
        Returns whether queue is full and all elements match """
        self.detections.appendleft(value)
        return value is not None and self.detections.count(value) == self.saved_detections

    def update(self,detected_code):
        """Updates FSM with latest code"""
        if detected_code is None:
            self.set_led(self.state, halved=True) #LED will be off-color if no hand/gesture is detected
        else:
            self.set_led(self.state)
            
        match self.state: #Note: Raspbian comes w/ python 3.11 which supports match-case statements
            case 0: #Idle; red led
                if self.update_queue(detected_code): #If queue is full of matching, non-None elements
                    self.state=1
                    self.detection_time = gt(rt_clock)
                    if len(detected_code) == 1:
                        self.sense.show_letter(detected_code, [128, 0, 0])
            case 1: 
                #Gesture detected; blue led
                #If gesture stops matching goes to case 0
                #If gesture matches for time_until_selected seconds, goes to case 2
                queue_matches = self.update_queue(detected_code)
                if not queue_matches:
                    self.state=0
                    self.sense.clear()
                elif queue_matches and gt(rt_clock)-self.detection_time > self.time_until_selected: #If gesture detected does not change for time_until_selected seconds, store it
                    if detected_code == "Open_Palm": #Open Palm confirms message
                        self.state=3
                    elif detected_code == "Middle_Finger": #???
                        self.state=4
                    else: #Otherwise store letter and iterate
                        self.state=2
                        self.result_string += detected_code
                        self.sense.show_letter(detected_code)
                    self.detection_time = gt(rt_clock)
            
            case 2: #Gesture confirmed. Shows green led for time_until_confirmed seconds
                if gt(rt_clock)-self.detection_time > self.time_until_confirmed:
                    self.state=0
                    self.sense.clear()
            case 3: #Word confirmed. Show onscreen and return to Idle
                self.sense.show_message(self.result_string)
                self.result_string = ""
                self.detections.clear()
                self.state=0
                
            case 4: #???
                self.sense.show_message(":(", 0.3, [255,0,0])
                self.result_string = ""
                self.detections.clear()
                self.state=0

        if debug:
            print(self.detections)

def get_fingercode(landmarks : list, threshold_thumb = 15, threshold = 50):
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
                if debug>1:
                    print("Points")
                    for joint in joints:
                        print(joint)
                    print("Vectors")
                    for i in range(3):
                        print(joints[i+1]-joints[i])
                vectors = [None, None, None]
                for i in range(3):
                    vectors[i] = (joints[i+1] - joints[i]) / np.linalg.norm(joints[i+1] - joints[i])
                if debug>1:
                    print("Normed vectors")
                    for vector in vectors:
                        print(vector)
                angles = [None, None]
                angles[0] = np.rad2deg(np.arccos(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0)))
                angles[1] = np.rad2deg(np.arccos(np.clip(np.dot(vectors[1], vectors[2]), -1.0, 1.0)))
                if debug>1:
                    print("Finger "+ld.fingers_dict[finger]+" has angles: "+str(angles))
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
            if debug>1:
                print("Fingertips")
                print(fingertips)
            distances = [None,None,None,None]
            for i in range(4):
                distances[i] = np.linalg.norm(fingertips[i+1]-fingertips[0])
            if debug>1:
                print("Distances")
                print(distances)
                print("Closest: "+ld.fingers_dict[np.argmin(distances)+1])
            if np.amin(distances) < threshold:
                return np.argmin(distances)+1
            else:
                return None
            
def get_angle(landmarks : list):
    """Estimates hand angle by measuring the angle between the wrist coordinates and the middle finger's root coordinates
    Returns:
    ~ 0º Facing left
    ~ 90º Facing down
    ~ -90º Facing up
    ~ |180º| Facing right
    Issue: May be improved by averaging middle and ring fingers' coordinates
    """
    if landmarks is not None:
        for detected_hand in landmarks:
            wrist_base = np.array([detected_hand.landmark[0].x,detected_hand.landmark[0].y,detected_hand.landmark[0].z])
            m_finger_base = np.array([detected_hand.landmark[9].x,detected_hand.landmark[9].y,detected_hand.landmark[9].z])
            
            palm_vector = m_finger_base - wrist_base
            if debug>1:
                print("Palm angle:")
                print(np.rad2deg(np.arctan2(palm_vector[1],palm_vector[0])))
            return np.rad2deg(np.arctan2(palm_vector[1],palm_vector[0]))
            
def get_direction(landmarks : list, half_sector_threshold = 30):
    """Returns the direction the hand is facing if its angle fits between the threshold of a given sector.
    Returns:
    None if no direction is clear
    0 if left
    1 if up
    2 if right
    3 if down
    """
    positive = False
    hand_angle = get_angle(landmarks)
    if hand_angle is not None:
        if hand_angle > 0:
            positive = True

        hand_angle = np.absolute(hand_angle)
        if hand_angle < 0 + half_sector_threshold:
            return 0
        elif hand_angle > 180 - half_sector_threshold:
            return 2
        elif hand_angle > 90 - half_sector_threshold and hand_angle < 90 + half_sector_threshold:
            if positive:
                return 3
            else:
                return 1
    return None

def get_thumb_palm_contact(landmarks : list, threshold = 20):
    """Returns whether thumb is in contact with the palm or not, by measuring the angle w/ the index's first joint
    Returns:
    True if under threshold
    False otherwise"""
    if landmarks is not None:
        for detected_hand in landmarks:
            thumb_base = np.array([detected_hand.landmark[1].x,detected_hand.landmark[1].y,detected_hand.landmark[1].z]) #Joint 1
            thumb_mid = np.array([detected_hand.landmark[3].x,detected_hand.landmark[3].y,detected_hand.landmark[3].z]) #Joint 3
            index_base = np.array([detected_hand.landmark[5].x,detected_hand.landmark[5].y,detected_hand.landmark[5].z]) #Joint 5
            
            thumb_vector = (thumb_mid - thumb_base) / np.linalg.norm(thumb_mid - thumb_base)
            palm_vector = (index_base - thumb_base) / np.linalg.norm(index_base - thumb_base)
            
            angle = np.rad2deg(np.arccos(np.clip(np.dot(thumb_vector, palm_vector), -1.0, 1.0)))
            if debug>1:
                print(f"Thumb-palm angle: {angle}")
            if angle > threshold:
                return False
            else:
                return True
    
   
if pi:
    picam2 = Picamera2()
    picam2.start()
    fsm = LEDFSM()
else:
    vid = cv2.VideoCapture(0)
    

while(True): 
      
    # Capture the video frame 
    # by frame
    if pi:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        ret, frame = vid.read()
        
    #Make frame non-writable to speed up passing process
    frame.flags.writeable = False
    results = mp_hands.process(frame)
    
    if show_video:
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
    if debug>2:
        start_time = timeit.default_timer()
    fingercode = get_fingercode(results.multi_hand_landmarks)  
    if debug>2:
        elapsed_fingercode = timeit.default_timer() - start_time
    contacts = get_contact(results.multi_hand_landmarks)
    if debug>2:
        elapsed_contacts = timeit.default_timer() - elapsed_fingercode - start_time
    direction = get_direction(results.multi_hand_landmarks)
    if debug>2:
        elapsed_direction = timeit.default_timer() - elapsed_contacts - elapsed_fingercode - start_time
    tp_contact = get_thumb_palm_contact(results.multi_hand_landmarks)
    if debug>2:
        elapsed_thumb_palm = timeit.default_timer() - elapsed_direction - elapsed_contacts - elapsed_fingercode - start_time
    detected_letter = gc([fingercode,contacts,direction]) #Get letter by calling the GestureClassifier
    if debug>2:
        elapsed_detection = timeit.default_timer()- elapsed_thumb_palm - elapsed_direction - elapsed_contacts - elapsed_fingercode - start_time
    if pi:
        fsm.update(detected_letter) #Update FSM with latest detection
    
    if debug:
        print(str(fingercode) + " - " + str(contacts)+ " - "+ ld.directions_dict[direction])
        print(detected_letter)
    if debug>2:
        print("Fingercode: "+str(elapsed_fingercode))
        print("Contacts: "+str(elapsed_contacts))
        print("Direction: "+str(elapsed_direction))
        print("Classifying: "+str(elapsed_detection))

    # Q - Quit program
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
if not pi:
    vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
