from ultralytics import YOLO

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load a pretrained model
    model = YOLO("yolov8n.pt")

    # Fine-tune on your dataset
    model.train(data="birdseye_dataset.yaml", epochs=100, imgsz=640)

    # Inference
    results = model("input_image.jpg")
    count = len(results[0].boxes.cls[results[0].boxes.cls == 0])  # Class 0 = person
    print(f"Total people: {count}")
