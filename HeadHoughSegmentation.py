import cv2
import numpy as np
import urllib.request


class HeadInstanceSegmentation:
    def __init__(self, model_weights, model_config, confidence_threshold=0.5):
        """
        Initialize with the paths to the Mask R-CNN weights and config files.
        The COCO-trained model assumes the 'person' class is labeled with ID 1.
        """
        self.net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)
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
        result = image.copy()

        for i in range(boxes.shape[2]):
            score = boxes[0, 0, i, 2]
            if score > self.confidence_threshold:
                class_id = int(boxes[0, 0, i, 1])
                # Only process detections for 'person'
                if class_id != 1:
                    continue

                # Extract bounding box and scale it to the image size.
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # For a birds‐eye view where only the head is visible,
                # you might optionally crop the bounding box to the upper portion.
                box_width = endX - startX
                box_height = endY - startY

                # Resize the corresponding mask to the bounding box dimensions.
                mask = masks[i, class_id]
                mask = cv2.resize(mask, (box_width, box_height))
                mask = (mask > 0.5)

                # Choose a random color for this person.
                color = [int(c) for c in np.random.randint(0, 255, size=3)]

                # Overlay the mask on the image: blend the mask color with the ROI.
                roi = result[startY:endY, startX:endX]
                roi[mask] = ((0.5 * np.array(color)) + (0.5 * roi[mask])).astype("uint8")

                # Optionally, draw contours along the edges of the mask.
                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, color, 2)
                result[startY:endY, startX:endX] = roi

        return result


# Example usage:
if __name__ == "__main__":
    # Replace these paths with the actual paths to your Mask R-CNN files.
    model_weights = "frozen_inference_graph.pb"


    model_config = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    image = cv2.imread(r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\0751.jpg")
    segmenter = HeadInstanceSegmentation(model_weights, model_config, confidence_threshold=0.6)
    result = segmenter.segment_heads(image)

    cv2.imshow("Head Instance Segmentation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
