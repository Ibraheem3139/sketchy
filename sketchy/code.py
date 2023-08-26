import cv2


def convert_to_pencil_sketch(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred = 255 - blurred

    # Create the pencil sketch
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    return pencil_sketch


if __name__ == "__main__":
    image_path = "ben.png"

    # Load the image
    image = cv2.imread(image_path)

    # Display the original image and allow user to select area
    cv2.imshow("Original Image", image)
    roi = cv2.selectROI("Original Image", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    # Crop the selected area
    selected_area = image[y:y + h, x:x + w]

    # Convert the selected area to pencil sketch
    pencil_sketch = convert_to_pencil_sketch(selected_area)

    # Display the original selected area and pencil sketch
    cv2.imshow("Selected Area", selected_area)
    cv2.imshow("Pencil Sketch", pencil_sketch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
