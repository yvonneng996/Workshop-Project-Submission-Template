import cv2
import os
import matplotlib.pyplot as plt
#import imutils
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from natsort import natsorted

import os
import numpy as np
from PIL import Image
import streamlit as st  # make sure to import Streamlit


def stitch_patches(binary_vessel, IDs):

    final_patch_dict = {}

    # -------------------------------------------------
    # STEP 1 — Extract patch index and store directly
    # -------------------------------------------------
    for patch, ID in zip(binary_vessel, IDs):

        filename = os.path.split(ID)[1]
        index = int(filename.partition("_")[2][:-4])

        # Decide crop type based on index position
        row = index // 9
        col = index % 9

        # -------- FIRST ROW (0–8) --------
        if row == 0:
            if col == 0:
                crop = patch[0:256, 0:256]
            else:
                crop = patch[0:256, 128:256]

        # -------- LEFT COLUMN --------
        elif col == 0:
            crop = patch[128:256, 0:256]

        # -------- NORMAL INNER PATCH --------
        else:
            crop = patch[128:256, 128:256]

        final_patch_dict[index] = crop.copy()

    # -------------------------------------------------
    # STEP 2 — Safety check
    # -------------------------------------------------
    missing = [i for i in range(63) if i not in final_patch_dict]
    if missing:
        print("Missing indices:", missing)
        raise ValueError("Some patches missing!")

    # -------------------------------------------------
    # STEP 3 — Stitch 7 × 9 grid
    # -------------------------------------------------
    rows = []

    for r in range(7):
        row_patches = []

        for c in range(9):
            idx = r * 9 + c
            row_patches.append(final_patch_dict[idx])

        row_img = np.hstack(row_patches)
        rows.append(row_img)

    final_image = np.vstack(rows)
    final_image = final_image[0:980, 0:1280]
    return final_image