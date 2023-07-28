import numpy as np
import os

from PIL import Image
import cv2

def match_portrait(probe_image: Image, template_images_payload: dict) -> str:
    # resize the probe image to match the template image, which is 128x128, but keep the aspect ratio
    # make sure the resized image is at least 128x128
    # resize wants width, height
    if probe_image.height > probe_image.width:
        resized_probe_image = probe_image.resize((128, int(128 * probe_image.height / probe_image.width)))
    else:
        resized_probe_image = probe_image.resize((int(128 * probe_image.width / probe_image.height), 128))
    # split by channel and convert to numpy array
    resized_probe_image = np.array(resized_probe_image)
    resized_probe_image_r = resized_probe_image[:, :, 0]
    resized_probe_image_g = resized_probe_image[:, :, 1]
    resized_probe_image_b = resized_probe_image[:, :, 2]
    #resized_probe_image_gray = cv2.cvtColor(resized_probe_image, cv2.COLOR_RGB2GRAY)

    rgb_match_values = {}
    #gray_match_values = {}
    method = cv2.TM_CCOEFF_NORMED
    for character_name in template_images_payload:
        template_image = template_images_payload[character_name]

        #template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        template_image_r = template_image[:, :, 0]
        template_image_g = template_image[:, :, 1]
        template_image_b = template_image[:, :, 2]

        # Perform match operations.
        match_result_r = cv2.matchTemplate(resized_probe_image_r, template_image_r, method)
        match_result_g = cv2.matchTemplate(resized_probe_image_g, template_image_g, method)
        match_result_b = cv2.matchTemplate(resized_probe_image_b, template_image_b, method)
        #match_result_gray = cv2.matchTemplate(resized_probe_image_gray, template_image_gray, method)

        # store the maximum match value in a dictionary
        # don't need to compute an average because the sum will have the same order
        rgb_match_values[character_name] = match_result_r[0][0] + match_result_g[0][0] + match_result_b[0][0]
        #gray_match_values[template_image_filename] = match_result_gray[0][0]

    # sort the dictionary by values
    sorted_rgb_match_values = sorted(rgb_match_values.items(), key=lambda x: x[1], reverse=True)
    #sorted_gray_match_values = sorted(gray_match_values.items(), key=lambda x: x[1], reverse=True)

    # return the top 1 result
    return sorted_rgb_match_values[0][0]