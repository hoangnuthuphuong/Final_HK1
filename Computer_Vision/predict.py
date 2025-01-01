import cv2
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime

def main():
    # Load model
    model = YOLO('runs/detect/train_v8ok/weights/best.pt')

    # Set up webcam
    cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Đặt chiều rộng khung hình
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Đặt chiều cao khung hình

    # Tạo thư mục lưu ảnh nếu chưa có
    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)

    print("Nhấn phím 's' để chụp và lưu ảnh. Nhấn 'ESC' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam.")
            break

        # Thực hiện dự đoán với YOLO
        results = model(frame)
        annotated_frame = results[0].plot()  # Vẽ các bounding box và nhãn

        # Hiển thị video trực tiếp
        cv2.imshow("HA PHUONG COMPUTER VISION", annotated_frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím 's' để chụp và lưu ảnh
        if key == ord('s'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Ảnh đã được lưu tại: {save_path}")

        # Nhấn ESC để thoát
        if key == 27:
            print("Thoát chương trình.")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


