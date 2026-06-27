import cv2
import os

OUTPUT_FOLDER = "static/outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # CLAHE
    # -------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    clahe_image = clahe.apply(gray)

    clahe_path = os.path.join(
        OUTPUT_FOLDER,
        "clahe.jpg"
    )

    cv2.imwrite(
        clahe_path,
        clahe_image
    )

    # -------------------------
    # Histogram
    # -------------------------

    histogram = cv2.equalizeHist(gray)

    histogram_path = os.path.join(
        OUTPUT_FOLDER,
        "histogram.jpg"
    )

    cv2.imwrite(
        histogram_path,
        histogram
    )

    # -------------------------
    # Green Channel
    # -------------------------

    green = image[:,:,1]

    green_path = os.path.join(
        OUTPUT_FOLDER,
        "green.jpg"
    )

    cv2.imwrite(
        green_path,
        green
    )

    # -------------------------
    # Gaussian Filter
    # -------------------------

    gaussian = cv2.GaussianBlur(
        green,
        (5,5),
        0
    )

    gaussian_path = os.path.join(
        OUTPUT_FOLDER,
        "gaussian.jpg"
    )

    cv2.imwrite(
        gaussian_path,
        gaussian
    )

    # -------------------------
    # Median Filter
    # -------------------------

    median = cv2.medianBlur(
        gaussian,
        5
    )

    median_path = os.path.join(
        OUTPUT_FOLDER,
        "median.jpg"
    )

    cv2.imwrite(
        median_path,
        median
    )

    return {

        "clahe":clahe_path,

        "histogram":histogram_path,

        "green":green_path,

        "gaussian":gaussian_path,

        "median":median_path

    }