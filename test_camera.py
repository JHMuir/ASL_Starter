"""
ASL (American Sign Language) Real-Time Recognition
===================================================
This script uses a webcam to capture hand gestures and predict ASL letters
using a trained CNN model with MediaPipe for hand detection.

Usage:
    python camera.py
    
Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
    
Requirements:
    pip install tensorflow opencv-python mediapipe
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import cv2
import numpy as np
import mediapipe as mp
import keras

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your trained model (update this!)
MODEL_PATH = 'asl_model_best.keras'  # or 'asl_model_final.keras'

# Detection confidence thresholds
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
PREDICTION_CONFIDENCE_THRESHOLD = 0.7

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ASL letters the model can recognize (J and Z excluded - they require motion)
ASL_LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]


# =============================================================================
# HAND DETECTOR CLASS
# =============================================================================

class ASLRecognizer:
    """Handles hand detection and ASL letter prediction."""
    
    def __init__(self, model_path):
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Store recent predictions for smoothing
        self.prediction_history = []
        self.history_size = 5
    
    def preprocess_hand_region(self, frame, bbox):
        """
        Extract and preprocess the hand region for model prediction.
        
        Args:
            frame: BGR image from camera
            bbox: Tuple of (x_min, y_min, x_max, y_max)
            
        Returns:
            Preprocessed image ready for model input, or None if processing fails
        """
        x_min, y_min, x_max, y_max = bbox
        
        try:
            # Crop hand region
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            if hand_region.size == 0:
                return None
            
            # Convert to grayscale (model expects single channel)
            hand_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size (28x28)
            hand_resized = cv2.resize(hand_gray, (28, 28))
            
            # Normalize pixel values to [0, 1]
            hand_normalized = hand_resized.astype('float32') / 255.0
            
            # Reshape for model: (batch_size, height, width, channels)
            hand_input = hand_normalized.reshape(1, 28, 28, 1)
            
            return hand_input
            
        except Exception as e:
            print(f"Error preprocessing hand region: {e}")
            return None
    
    def get_hand_bbox(self, hand_landmarks, frame_shape):
        """
        Calculate bounding box around detected hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Tuple of (height, width, channels)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) with padding
        """
        h, w, _ = frame_shape
        
        # Get x and y coordinates of all landmarks
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding (15% of bounding box size)
        padding_x = int(0.15 * (x_max - x_min))
        padding_y = int(0.15 * (y_max - y_min))
        
        # Ensure coordinates stay within frame bounds
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return (x_min, y_min, x_max, y_max)
    
    def predict_letter(self, hand_input):
        """
        Predict ASL letter from preprocessed hand image.
        
        Args:
            hand_input: Preprocessed image array
            
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        # Get model prediction
        prediction = self.model.predict(hand_input, verbose=0)
        
        # Get class with highest probability
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)
        
        # Apply confidence threshold
        if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
            predicted_letter = ASL_LETTERS[predicted_index]
        else:
            predicted_letter = "?"
        
        return predicted_letter, confidence
    
    def smooth_prediction(self, letter, confidence):
        """
        Smooth predictions using recent history to reduce flickering.
        
        Args:
            letter: Current predicted letter
            confidence: Current prediction confidence
            
        Returns:
            Smoothed letter prediction
        """
        # Add to history
        self.prediction_history.append((letter, confidence))
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Count occurrences weighted by confidence
        letter_scores = {}
        for hist_letter, hist_conf in self.prediction_history:
            if hist_letter != "?":
                letter_scores[hist_letter] = letter_scores.get(hist_letter, 0) + hist_conf
        
        # Return letter with highest weighted score
        if letter_scores:
            return max(letter_scores, key=letter_scores.get)
        return "?"
    
    def process_frame(self, frame):
        """
        Process a single frame: detect hand and predict ASL letter.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Annotated frame with prediction
        """
        # Flip for mirror view (more intuitive for user)
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                bbox = self.get_hand_bbox(hand_landmarks, frame.shape)
                x_min, y_min, x_max, y_max = bbox
                
                # Preprocess and predict
                hand_input = self.preprocess_hand_region(frame, bbox)
                
                if hand_input is not None:
                    letter, confidence = self.predict_letter(hand_input)
                    smoothed_letter = self.smooth_prediction(letter, confidence)
                    
                    # Draw prediction text
                    label = f"{smoothed_letter} ({confidence:.0%})"
                    cv2.putText(
                        frame, label,
                        (x_min, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2
                    )
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame
    
    def cleanup(self):
        """Release resources."""
        self.hands.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please train the model first using asl_training.py")
        print("Or update MODEL_PATH to point to your trained model.")
        return
    
    # Initialize recognizer
    recognizer = ASLRecognizer(MODEL_PATH)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n" + "=" * 50)
    print("ASL Recognition Started!")
    print("=" * 50)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 50 + "\n")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            
            # Process frame
            annotated_frame = recognizer.process_frame(frame)
            
            # Display frame
            cv2.imshow('ASL Recognition', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
    
    finally:
        # Cleanup
        recognizer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()