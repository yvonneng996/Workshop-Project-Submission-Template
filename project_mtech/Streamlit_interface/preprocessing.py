import cv2
import numpy as np
import os
import model_unet
import model_swunet
def process_images(orig_img, mask_img):

    # Keep original color image
    orig = orig_img.copy()

    # Convert mask to grayscale ONLY for masking
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    # Resize mask if needed
    if mask_gray.shape[:2] != orig.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (orig.shape[1], orig.shape[0]))

    # Convert to proper mask format (0 or 255)
    mask_bin = np.where(mask_gray > 0, 255, 0).astype(np.uint8)

    # Apply mask — RESULT STAYS COLORED
    result = cv2.bitwise_and(orig, orig, mask=mask_bin)
    #norm = np.zeros((256,256))
    norm_image = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX)
    img_h, img_w, _ = (980,1280,3)

    split_width = 256
    split_height = 256
    overlap = 0.5

    # ---- START POINT LOGIC (embedded, no extra def) ----
    X_points = [0]
    stride_x = int(split_width * (1 - overlap))
    counter = 1
    while True:
        pt = stride_x * counter
        if pt + split_width >= img_w:
            if split_width != img_w:
                X_points.append(img_w - split_width)
            break
        else:
            X_points.append(pt)
        counter += 1

    Y_points = [0]
    stride_y = int(split_height * (1 - overlap))
    counter = 1
    while True:
        pt = stride_y * counter
        if pt + split_height >= img_h:
            if split_height != img_h:
                Y_points.append(img_h - split_height)
            break
        else:
            Y_points.append(pt)
        counter += 1
    # ------------------------------------------------------

    # ---- SPLITTING ----
    d = 0
    patches=[]
    for i in Y_points:
        for j in X_points:
            split = norm_image[i:i+split_height, j:j+split_width]
            #clahe=cv2.createCLAHE(clipLimit=5.0,tileGridSize=(8,8))
            (b,g,r)=cv2.split(split)
            #final_imgb = clahe.apply(b)
            #final_imgg = clahe.apply(g)
            #final_imgr = clahe.apply(r)
            merge_img = cv2.merge([b,g,r])
            patches.append(merge_img)
            
    return patches