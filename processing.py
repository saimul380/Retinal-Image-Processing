import cv2
import os
import time
import numpy as np
from skimage.feature import graycomatrix, graycoprops

OUTPUT_FOLDER = "static/outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_image(image_path):

    start_time = time.time()

    # -------------------------
    # Read Image
    # -------------------------
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    # Gray Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Green Channel
    # -------------------------
    green = image[:, :, 1]

    green_path = os.path.join(OUTPUT_FOLDER, "green.jpg")
    cv2.imwrite(green_path, green)

    # -------------------------
    # CLAHE
    # -------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(green)

    clahe_path = os.path.join(OUTPUT_FOLDER, "clahe.jpg")
    cv2.imwrite(clahe_path, clahe_image)

    # -------------------------
    # Histogram Equalization
    # -------------------------
    histogram = cv2.equalizeHist(green)

    histogram_path = os.path.join(OUTPUT_FOLDER, "histogram.jpg")
    cv2.imwrite(histogram_path, histogram)

    # -------------------------
    # Gaussian Filter
    # -------------------------
    gaussian = cv2.GaussianBlur(
        clahe_image,
        (5, 5),
        0
    )

    gaussian_path = os.path.join(OUTPUT_FOLDER, "gaussian.jpg")
    cv2.imwrite(gaussian_path, gaussian)

    # -------------------------
    # Median Filter
    # -------------------------
    median = cv2.medianBlur(
        gaussian,
        5
    )

    median_path = os.path.join(OUTPUT_FOLDER, "median.jpg")
    cv2.imwrite(median_path, median)

    # -------------------------
    # Morphological Opening
    # -------------------------
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

    # -------------------------
    # Morphological Closing
    # -------------------------
    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    closing_path = os.path.join(OUTPUT_FOLDER, "closing.jpg")
    cv2.imwrite(closing_path, closing)

    # -------------------------
    # Adaptive Threshold
    # -------------------------
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

    # -------------------------
    # Blood Vessel Segmentation
    # -------------------------
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
    # Blood Vessel Overlay
    # -------------------------
    overlay = image.copy()

    overlay[vessel > 0] = (0, 0, 255)

    overlay = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    overlay_path = os.path.join(
        OUTPUT_FOLDER,
        "overlay.jpg"
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

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
    
    # Additional Features
    min_intensity = int(vessel.min())
    max_intensity = int(vessel.max())
    median_intensity = round(float(np.median(vessel)), 2)
    variance = round(float(vessel.var()), 2)
    rms_intensity = round(
        float(np.sqrt(np.mean(vessel.astype(np.float32) ** 2))),
        2
    )
    black_pixels = total_pixels - vessel_pixels
    black_density = round(
        (black_pixels / total_pixels) * 100,
        2
    )

    # -------------------------
    # GLCM Texture Features
    # -------------------------
    glcm = graycomatrix(
        vessel,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = round(
        float(graycoprops(glcm, 'contrast')[0, 0]),
        4
    )
    energy = round(
        float(graycoprops(glcm, 'energy')[0, 0]),
        4
    )
    homogeneity = round(
        float(graycoprops(glcm, 'homogeneity')[0, 0]),
        4
    )
    correlation = round(
        float(graycoprops(glcm, 'correlation')[0, 0]),
        4
    )
    dissimilarity = round(
        float(graycoprops(glcm, 'dissimilarity')[0, 0]),
        4
    )

    features = {
        "Mean Intensity": mean_intensity,
        "Standard Deviation": std_intensity,
        "Minimum Intensity": min_intensity,
        "Maximum Intensity": max_intensity,
        "Median Intensity": median_intensity,
        "Variance": variance,
        "RMS Intensity": rms_intensity,
        "Vessel Pixels": vessel_pixels,
        "Total Pixels": total_pixels,
        "Vessel Density (%)": vessel_density,
        "Background Pixels": black_pixels,
        "Background Density (%)": black_density,
        "GLCM Contrast": contrast,
        "GLCM Energy": energy,
        "GLCM Homogeneity": homogeneity,
        "GLCM Correlation": correlation,
        "GLCM Dissimilarity": dissimilarity
    }

    # -------------------------
    # Processing Time
    # -------------------------
    end_time = time.time()

    processing_time = round(
        end_time - start_time,
        3
    )

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
        "overlay": overlay_path,
        "features": features,
        "processing_time": processing_time
    }