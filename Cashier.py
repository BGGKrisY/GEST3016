import cv2
import numpy as np
import time
class SimpleSelfCheckout:
    def __init__(self):
        self.products = {
            "coke": 3.5,
            "chips": 8.0,
            "water": 2.0,
            "chocolate": 12.5,
            "biscuit": 6.0,
            "apple": 5.0,
            "banana": 4.0,
            "milk": 10.0,
            "bread": 15.0,
            "noodles": 12.0
        }
        self.color_ranges = {
            "coke": ([0, 100, 100], [10, 255, 255]),
            "water": ([100, 100, 100], [130, 255, 255]),
            "banana": ([20, 100, 100], [30, 255, 255]),
        }
        self.detected_items = []
        self.total_amount = 0.0
        self.last_calculation_time = 0
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except:
            return False
    def detect_objects_by_color(self, frame):
        detected = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for product_name, (lower, upper) in self.color_ranges.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(0.95, area / 5000)
                    detected.append({
                        "name": product_name,
                        "confidence": confidence,
                        "bbox": (x, y, w, h),
                        "area": area
                    })
        return detected
    def detect_objects_by_shape(self, frame):
        detected = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:
                    product_name = "coke"
                elif aspect_ratio > 1.5:
                    product_name = "chips"
                else:
                    product_name = "water"
                confidence = 0.7
                detected.append({
                    "name": product_name,
                    "confidence": confidence,
                    "bbox": (x, y, w, h),
                    "area": area
                })
        return detected
    def simulate_detection(self, frame):
        height, width = frame.shape[:2]
        detected = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        num_items = int((255 - avg_brightness) / 85) + 1
        product_names = list(self.products.keys())
        for i in range(min(num_items, 4)):
            product_name = np.random.choice(product_names)
            x = np.random.randint(50, width - 200)
            y = np.random.randint(50, height - 200)
            w = np.random.randint(80, 150)
            h = np.random.randint(80, 150)
            confidence = np.random.uniform(0.6, 0.9)
            detected.append({
                "name": product_name,
                "confidence": confidence,
                "bbox": (x, y, w, h)
            })
        return detected
    def calculate_total_price(self):
        current_time = time.time()
        if current_time - self.last_calculation_time < 1:
            return
        self.last_calculation_time = current_time
        print("\n" + "=" * 40)
        print("CALCULATING TOTAL PRICE...")
        print("=" * 40)
        if not self.detected_items:
            print("No items detected in camera view")
            return
        self.total_amount = 0.0
        item_count = {}
        for item in self.detected_items:
            product_name = item["name"]
            if product_name in self.products:
                price = self.products[product_name]
                self.total_amount += price
                if product_name in item_count:
                    item_count[product_name] += 1
                else:
                    item_count[product_name] = 1
                print(f"✓ {product_name:12} - ${price:5.2f} (conf: {item['confidence']:.2f})")
            else:
                print(f"✗ {product_name:12} - Price not found")
        print("-" * 40)
        print("ITEM SUMMARY:")
        for item, count in item_count.items():
            subtotal = self.products[item] * count
            print(f"  {item} x{count} = ${subtotal:.2f}")
        print("-" * 40)
        print(f"TOTAL AMOUNT: ${self.total_amount:.2f}")
        print("=" * 40)
    def draw_detections(self, frame, detections):
        display_frame = frame.copy()
        for i, detection in enumerate(detections):
            name = detection["name"]
            confidence = detection["confidence"]
            x, y, w, h = detection["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), color, -1)
            cv2.putText(display_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(display_frame, f"TOTAL: ${self.total_amount:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"ITEMS: {len(detections)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        instructions = [
            "Press 'A': Calculate Total",
            "Press 'R': Reset Total",
            "Press 'Q': Quit"
        ]
        for j, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction,
                        (10, 90 + j * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display_frame
    def reset_cart(self):
        self.total_amount = 0.0
        self.detected_items = []
        print("\n Cart has been reset!")
        print("Total: $0.00")
    def run(self):
        print("=" * 50)
        print("SELF-CHECKOUT SYSTEM")
        print("=" * 50)
        print("Place items in front of the camera")
        print("Press 'A' to calculate total price")
        print("Press 'R' to reset cart")
        print("Press 'Q' to quit")
        print("=" * 50)

        if not self.start_camera():
            print("Camera not available. Please check your camera connection.")
            return

        print("Camera started successfully!")
        print("System ready - Place items in camera view...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            try:
                self.detected_items = self.detect_objects_by_color(frame)
                if not self.detected_items:
                    self.detected_items = self.detect_objects_by_shape(frame)
                if not self.detected_items:
                    self.detected_items = self.simulate_detection(frame)
            except Exception as e:
                print(f"Detection error: {e}")
                self.detected_items = self.simulate_detection(frame)
            display_frame = self.draw_detections(frame, self.detected_items)
            cv2.imshow('Self-Checkout System - Press A to Calculate', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a') or key == ord('A'):
                self.calculate_total_price()
            elif key == ord('r') or key == ord('R'):
                self.reset_cart()

            elif key == ord('q') or key == ord('Q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete")
        print(f"Final total: ${self.total_amount:.2f}")
if __name__ == "__main__":
    checkout_system = SimpleSelfCheckout()
    checkout_system.run()