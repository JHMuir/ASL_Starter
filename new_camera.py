import cv2
import mediapipe as mp
# import numpy
# import keras

# C++ -> Binary -> PC 
# Python -> C > Binary -> PC 

# Keras -> Tensorflow -> PC

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# class ASL_Recognizer:
#     def __init__(self, model):
#         self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
#         # The AI Model
#         self.model = keras.models.load_model(model)
#         # The mp Hand stuff
#         self.mp_hands = None
#         self.hands = None
#         self.mp_draw = None
        
#         # Our local stored predictions
#         self.prediction_history = []
#         self.history_size = 5
        
#     def predict_letter(self, hand_input):
#         prediction = self.model.predict(hand_input)
#         predicted_index = numpy.argmax(prediction)
#         predicted_letter = self.alphabet[predicted_index]
#         return predicted_letter
        

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw hand landmarks and make predictions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding box around the hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            h, w, c = frame.shape
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Add dynamic padding to the bounding box
            padding = int(0.1 * (x_max - x_min))  # 10% of the bounding box width
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop and preprocess the hand region
            try:
                hand_region = frame[y_min:y_max, x_min:x_max]
                hand_region_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                hand_region_resized = cv2.resize(hand_region_gray, (28, 28))  # Resize to model input size
                hand_region_normalized = hand_region_resized / 255.0  # Normalize pixel values
                hand_region_input = hand_region_normalized.reshape(1, 28, 28, 1)  # Reshape for the model

                # Predict the gesture
                # prediction = model.predict(hand_region_input)
                # confidence = np.max(prediction)
                # if confidence > 0.7:  # Use a confidence threshold
                #     predicted_letter = letterpred[np.argmax(prediction)]
                # else:
                #     predicted_letter = "Unknown"

                # # Display the predicted letter on the frame
                # cv2.putText(
                #     frame, 
                #     f"Prediction: {predicted_letter}", 
                #     (x_min, y_min - 10), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     1, 
                #     (0, 255, 0), 
                #     2
                # )
            except Exception as e:
                print(f"Error processing hand region: {e}")

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Test Camera", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()