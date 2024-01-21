fingers_dict = {0:"Pulgar", 1:"Indice", 2:"Corazon",3:"Anular",4:"Menique"}
directions_dict = {None: "", 0:"Izq", 1:"Ar", 2:"De", 3:"Ab"}

class CharacterDescriptor():
    """Describes all known characters based on 3 descriptors: ( ExtendedFingers, Contacts, Direction )
    ExtendedFingers - List of 5 entries, 1 if finger is extended, starting from thumb
    Contacts - Int from -1, 1 to 4, -1 means no contacts, 1-4 is the finger contacting the thumb
    Direction - Int from 0 to 4, specifies whether the palm faces Up or Down, Left or Right. Left & Right are treated the same.
    If None, that descriptor is irrelevant for this Character's detection"""
    
    A = ([1,0,0,0,0],None,0) #LSE, puño horizontal con el pulgar extendido
    B = ([1,1,1,1,1],None,0) #LSE, palma horizontal
    C = (None,None,None)
    D = ([1,1,0,1,1],2,None) #LSE, Contacto P-C, dedos restantes extendidos
    E = ([0,0,0,0,0],4,1) #ASL, puño cerrado vertical, Contacto P-M
    F = ([1,0,1,1,1],1,1) #ASL, contacto P-I, vertical, dedos restantes extendidos
    G = ([0,1,0,0,0],None,0) #LSE, ind extendido horizontal(Apuntar)
    H = ([0,1,1,0,0],None,0) #LSE, ind + cor extendidos horizontales (Pistola)
    I = ([0,0,0,0,1],None,0) #LSE*, men extendido horizontal
    J = ([0,0,0,0,1],None,1) #ASL*, men extendido vertical
    K = ([0,1,0,0,0],None,1) #ASL*, ind extendido vertical (Pregunta)
    L = ([1,1,0,0,0],None,None) #ASL, ind + pulgar extendidos vertical (Lerdo)
    M = ([0,1,1,1,0],None,3) #LSE, ind + cor+ an extendidos hacia abajo
    N = ([0,1,1,0,0],None,3) #LSE, ind + cor extendidos hacia abajo 
    O = ([1,0,1,1,1],1,0) #LSE, contacto P-I, horizontal, dedos restantes extendidos (OK)
    P = ([1,1,1,1,0],None,None) #LSE*, men recogido (Baphomet)
    Q = ([0,1,0,0,1],None,None) #LSE, men + ind extendidos (Cuernos)
    R = ([1,1,0,1,1],None,None) #LSE, cor recogido (MiddleFingern't)
    S = ([1,1,0,0,0],1,None) #LSE*/ASL*, contacto P-I, resto recogidos
    T = (None,None,None)
    U = ([1,1,1,0,0],None,1) #ASL*, cor + ind + pul extendidos vertical
    V = ([0,1,1,0,0],None,1) #LSE/ASL, ind + cor extendidos hacia arriba (Victoria)
    W = ([0,1,1,1,0],None,1) #ASL, ind + cor + an extendidos hacia arriba
    X = (None,None,None)
    Y = ([1,0,0,0,1],None,None) #ASL, pul + men extendidos (RocknRoll)
    Z = ([1,1,0,0,1],None,None) #LSE*/ASL*, pul + ind + men extendidos (Spiderman) 
    Middle_Finger = ([0,0,1,0,0],None,1)
    Open_Palm = ([1,1,1,1,1],None,1)
    
class GestureClassifier():
    
    @staticmethod
    def list_to_number(finger_list:list):
        """Interprets list values as a binary number & returns it"""
        return finger_list[0]*16 + finger_list[1]*8 + finger_list[2]*4 + finger_list[3]*2 + finger_list[4]
    
    def create_classifier_dict(self):
        """Fills the __classifier__ dict using the class instances
        Using a dict instead of the CharacterDescriptor class would probably be more Pythonic & Efficient but adding, modifying & documenting
        gestures seems easier & less error-prone like this"""
        for code, value in vars(CharacterDescriptor).items():
            if not code.startswith("__") and not value[0] is None:
                finger_number = self.list_to_number(value[0])
                if finger_number in self.__classifier__:   
                    if type(self.__classifier__[finger_number]) == str:
                        #print("Found match between "+self.__classifier__[finger_number]+" and "+code)
                        stored_value = self.__classifier__[finger_number]
                        self.__classifier__[finger_number] = {}
                        self.__classifier__[finger_number][vars(CharacterDescriptor)[stored_value][2]] = stored_value
                    if value[2] in self.__classifier__[finger_number]:
                        if type(self.__classifier__[finger_number][value[2]]) == str:
                            print("Found match between "+self.__classifier__[finger_number][value[2]]+" and "+code)
                            stored_value = self.__classifier__[finger_number][value[2]]
                            self.__classifier__[finger_number][value[2]] = {}
                            self.__classifier__[finger_number][value[2]][vars(CharacterDescriptor)[stored_value][1]] = stored_value
                        self.__classifier__[finger_number][value[2]][value[1]] = code   
                    else:
                        self.__classifier__[finger_number][value[2]] = code   
                else:
                    self.__classifier__[finger_number] = code
        
    def __init__(self):
        self.__classifier__ = {}
        self.create_classifier_dict()
        print(self.__classifier__)
        
    def __call__(self, hand_info):
        """Compares given hand_info to the __classifier__ dict object. Returns matched character, or None"""
        if hand_info[0] is not None:
            hand_info[0] = self.list_to_number(hand_info[0])
            hand_info.append(hand_info.pop(1)) # Reorder list to match the __classifier__ dict
            if hand_info[1] == 2:
                hand_info[1] = 0 # Right or left are indifferent
            i = 0
            result = self.__classifier__
            while True:
                result_aux = result.get(hand_info[i])
                if type(result_aux) == str:
                    break
                elif type(result_aux) == dict:
                    i += 1
                else:
                    result_aux = result.get(None)
                    if type(result_aux) == dict:
                        i += 1
                    else:
                        break
                result = result_aux
            return result_aux