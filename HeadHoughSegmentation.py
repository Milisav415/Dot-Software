import cv2
import numpy as np


class HeadInstanceSegmentation:
    def __init__(self, model_weights, model_config, confidence_threshold=0.5):
        """
        Initialize with the paths to the Mask R-CNN weights and config files.
        The COCO-trained model assumes the 'person' class is labeled with ID 1.
        """
        self.net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)
        if not self.net.getLayerNames():
            print("Error: No layers found. The model might not be loaded correctly.")
        else:
            print("Network loaded successfully. Number of layers:", len(self.net.getLayerNames()))
        self.confidence_threshold = confidence_threshold

    def segment_heads(self, image):
        """
        Processes the image to detect persons, extract their segmentation masks,
        and overlay a unique color on each detected head.

        Parameters:
            image (numpy.ndarray): The input image in BGR format.

        Returns:
            result (numpy.ndarray): The image with each detected person colored and outlined.
        """
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.net.setInput(blob)

        # Forward pass to get detection boxes and masks.
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        print("Detections shape:", boxes.shape)  # Debug: Print shape of detections

        result = image.copy()
        detection_count = 0

        for i in range(boxes.shape[2]):
            score = boxes[0, 0, i, 2]
            print("Detection {}: score = {}".format(i, score))  # Debug: Show each detection's score
            if score > self.confidence_threshold:
                detection_count += 1
                class_id = int(boxes[0, 0, i, 1])
                if class_id != 1:  # Only process 'person'
                    continue

                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                box_width = endX - startX
                box_height = endY - startY

                mask = masks[i, class_id]
                mask = cv2.resize(mask, (box_width, box_height))
                mask = (mask > 0.5)

                color = [int(c) for c in np.random.randint(0, 255, size=3)]

                roi = result[startY:endY, startX:endX]
                roi[mask] = ((0.5 * np.array(color)) + (0.5 * roi[mask])).astype("uint8")

                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, color, 2)
                result[startY:endY, startX:endX] = roi

        print("Total detections above threshold:", detection_count)  # Debug info
        return result


# Example usage:
if __name__ == "__main__":
    # Replace these paths with the actual paths to your Mask R-CNN files.
    model_weights = "frozen_inference_graph.pb"
    model_config = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    image = cv2.imread(r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\0751.jpg")
    segmenter = HeadInstanceSegmentation(model_weights, model_config, confidence_threshold=0.1)
    result = segmenter.segment_heads(image)

    cv2.imshow("Head Instance Segmentation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
