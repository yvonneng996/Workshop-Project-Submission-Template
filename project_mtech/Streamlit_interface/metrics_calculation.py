import numpy as np
import cv2

def calculate_vessel_density(mask_img, vessel_img):

    # Convert to grayscale if RGB
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    if len(vessel_img.shape) == 3:
        vessel_img = cv2.cvtColor(vessel_img, cv2.COLOR_BGR2GRAY)

    vessel = np.where(vessel_img > 127, 255, 0)
    masking = np.where(mask_img > 127, 255, 0)

    masking1 = masking / 255
    area_of_region = (vessel / 255) * masking1

    a = np.sum(area_of_region == 1)
    c = np.sum(masking1 == 1)

    white_pixel = a / c if c != 0 else 0

    return white_pixel