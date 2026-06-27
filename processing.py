import cv2
import os

OUTPUT_FOLDER = "static/outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_image(image_path):

    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    # Gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Green channel
    green = image[:, :, 1]

    green_path = os.path.join(OUTPUT_FOLDER, "green.jpg")
    cv2.imwrite(green_path, green)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(green)

    clahe_path = os.path.join(OUTPUT_FOLDER, "clahe.jpg")
    cv2.imwrite(clahe_path, clahe_image)

    # Histogram Equalization
    histogram = cv2.equalizeHist(green)

    histogram_path = os.path.join(OUTPUT_FOLDER, "histogram.jpg")
    cv2.imwrite(histogram_path, histogram)

    # Gaussian Filter
    gaussian = cv2.GaussianBlur(
        clahe_image,
        (5, 5),
        0
    )

    gaussian_path = os.path.join(OUTPUT_FOLDER, "gaussian.jpg")
    cv2.imwrite(gaussian_path, gaussian)

    # Median Filter
    median = cv2.medianBlur(
        gaussian,
        5
    )

    median_path = os.path.join(OUTPUT_FOLDER, "median.jpg")
    cv2.imwrite(median_path, median)

    # Morphological Opening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    opening = cv2.morphologyEx(
        median,
        cv2.MORPH_OPEN,
        kernel
    )

    opening_path = os.path.join(OUTPUT_FOLDER, "opening.jpg")
    cv2.imwrite(opening_path, opening)

    # Morphological Closing
    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    closing_path = os.path.join(OUTPUT_FOLDER, "closing.jpg")
    cv2.imwrite(closing_path, closing)

    # Adaptive Threshold
    threshold = cv2.adaptiveThreshold(
        closing,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )

    threshold_path = os.path.join(OUTPUT_FOLDER, "threshold.jpg")
    cv2.imwrite(threshold_path, threshold)

    # Vessel Segmentation
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    vessel = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel2
    )

    vessel = cv2.morphologyEx(
        vessel,
        cv2.MORPH_CLOSE,
        kernel2
    )

    vessel_path = os.path.join(OUTPUT_FOLDER, "vessel.jpg")
    cv2.imwrite(vessel_path, vessel)

    # -------------------------
    # Feature Extraction
    # -------------------------

    mean_intensity = round(float(vessel.mean()), 2)
    std_intensity = round(float(vessel.std()), 2)
    vessel_pixels = int(cv2.countNonZero(vessel))
    total_pixels = vessel.shape[0] * vessel.shape[1]

    vessel_density = round(
        (vessel_pixels / total_pixels) * 100,
        2
    )

    features = {
        "Mean Intensity": mean_intensity,
        "Standard Deviation": std_intensity,
        "Vessel Pixels": vessel_pixels,
        "Total Pixels": total_pixels,
        "Vessel Density (%)": vessel_density
    }

    return {
        "clahe": clahe_path,
        "histogram": histogram_path,
        "green": green_path,
        "gaussian": gaussian_path,
        "median": median_path,
        "opening": opening_path,
        "closing": closing_path,
        "threshold": threshold_path,
        "vessel": vessel_path,
        "features": features
    }