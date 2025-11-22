import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import time
class UltraFastFoodCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltraFastFoodCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_food_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint = torch.load(model_path, map_location=device)
    print("Model file loaded successfully!")
    for key, value in checkpoint.items():
        if 'classifier.1.weight' in key:
            num_classes = value.shape[0]
            print(f"Detected model has {num_classes} classes")
            break
    model = UltraFastFoodCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model, num_classes, device
def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return None
    print("Camera started successfully!")
    return cap
def preprocess_frame(frame, img_size=128):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch
def predict_food(model, frame, device, class_names):
    with torch.no_grad():
        input_batch = preprocess_frame(frame)
        input_batch = input_batch.to(device)
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        return class_names[predicted_idx.item()], confidence.item()
def draw_prediction(frame, prediction, confidence, fps):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    if confidence > 0.8:
        color = (0, 255, 0)
    elif confidence > 0.6:
        color = (0, 255, 255)
    else:
        color = (0, 165, 255)
    cv2.putText(frame, f"Food: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bar_width = 200
    bar_height = 20
    bar_x = (width - bar_width) // 2
    bar_y = height - 40
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (100, 100, 100), -1)
    fill_width = int(bar_width * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 2)
    return frame
def main():
    model_path = "best_food_model_ultra_fast.pth"
    try:
        print("Loading model...")
        model, num_classes, device = load_food_model(model_path)
        class_names = [
            "Bread", "Dairy", "Dessert", "Egg", "Fried Food",
            "Meat", "Noodles", "Rice", "Seafood", "Soup",
            "Vegetables"
        ]
        if len(class_names) != num_classes:
            print(
                f"Warning: Class count mismatch! Model has {num_classes} classes, but {len(class_names)} names provided")
            class_names = [f"Food_{i + 1}" for i in range(num_classes)]
            print(f"Using auto-generated class names: {class_names}")
        else:
            print(f"Using class names: {class_names}")
        cap = setup_camera()
        if cap is None:
            return
        print("\n=== Food Recognition System ===")
        print("Controls:")
        print("Press 'q' - Quit program")
        print("Press 's' - Save current image")
        print("Press 'c' - Show class list")
        print("Press 'r' - Reload model")
        print("\nStarting real-time food recognition...")
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        prediction = "Aim at food..."
        confidence = 0.0
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break
            if frame_count % 3 == 0:
                try:
                    prediction, confidence = predict_food(model, frame, device, class_names)
                except Exception as e:
                    prediction = "Recognition Error"
                    confidence = 0.0
                    print(f"Prediction error: {e}")
            frame_count += 1
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = current_time
            frame = draw_prediction(frame, prediction, confidence, fps)
            cv2.imshow('Smart Food Recognition System - Press Q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"food_{prediction}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
            elif key == ord('c'):
                print("\n=== Recognizable Food Classes ===")
                for i, name in enumerate(class_names):
                    print(f"  {i + 1:2d}. {name}")
            elif key == ord('r'):
                print("Reloading model...")
                model, num_classes, device = load_food_model(model_path)
        cap.release()
        cv2.destroyAllWindows()
        print("Program exited")
    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()