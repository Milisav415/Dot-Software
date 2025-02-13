import cv2
import numpy as np
from torchvision import transforms

class DotingPeople:
    def __init__(self, dp=1.2, min_dist=50, param1=50, param2=30, min_radius=10, max_radius=30):
        """
        Initialize parameters for the Hough Circle Transform.

        dp: Inverse ratio of the accumulator resolution to the image resolution.
        min_dist: Minimum distance between the centers of detected circles.
        param1: Higher threshold for the Canny edge detector.
        param2: Accumulator threshold for circle detection (lower means more false circles).
        min_radius & max_radius: Range of circle radii to detect.
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect_heads(self, image):
        """
        Detect circular shapes (representing heads) in the provided image segment.

        Parameters:
            image (numpy.ndarray): The input image segment in BGR format.

        Returns:
            circles (numpy.ndarray or None): An array of detected circles, each defined by (x, y, r).
        """
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur or any other pre-processing as needed
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, self.dp, self.min_dist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0, :]  # Return array of circles (each as [x, y, r])
        return None


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise using a bilateral filter
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Optional: Apply a Gaussian blur (or use as alternative to bilateral filtering)
    blurred = cv2.GaussianBlur(denoised, (9, 9), 2)

    # Optional: Sharpen the image to enhance edges
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)

    return sharpened

def segment_and_mark_heads(original_image, preprocessed_image, tile_width, tile_height, detector):
    """
    Segments the image into tiles, detects heads in each tile (using the preprocessed image),
    and draws a dot on each detected head in the original image.

    Parameters:
        original_image (numpy.ndarray): The full original color image.
        preprocessed_image (numpy.ndarray): The preprocessed (typically grayscale) image.
        tile_width (int): Width of each tile.
        tile_height (int): Height of each tile.
        detector (DotingPeople): An instance of the head detector.

    Returns:
        output (numpy.ndarray): The original image with dots marking detected heads.
    """
    output = original_image.copy()
    h, w = original_image.shape[:2]
    dot_count = 0
    for y in range(0, h, tile_height):
        for x in range(0, w, tile_width):
            # Define the segment boundaries ensuring we do not exceed the image dimensions.
            proc_segment = preprocessed_image[y:min(y + tile_height, h), x:min(x + tile_width, w)]
            circles = detector.detect_heads(proc_segment)
            if circles is not None:
                for circle in circles:
                    cx, cy, r = circle
                    # Adjust circle coordinates to the original image by adding segment offsets.
                    center = (x + cx, y + cy)
                    # Draw a small dot at the center of each detected head.
                    cv2.circle(output, center, 2, (0, 0, 255), thickness=-1)
                    dot_count += 1

    print("Number of dots detected:", dot_count)
    return output

# Example usage:
if __name__ == "__main__":
    # Load the image.
    image = cv2.imread(r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\3240.jpg")
    if image is None:
        raise ValueError("Image not found. Please check the path.")

    tile_width = 100
    tile_height = 100

    # Create the preprocessed image.
    preprocessed = preprocess_image(image)

    # Create an instance of the detector.
    detector = DotingPeople(dp=1.5, min_dist=40, param1=100, param2=20, min_radius=10, max_radius=30)

    # Process the image: perform detection on the preprocessed image and draw dots on the original.
    result = segment_and_mark_heads(image, preprocessed, tile_width, tile_height, detector)

    cv2.imshow("Segmented Head Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

