from collections import Counter
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from PIL import Image
import cv2

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage import measure
from skimage import filters

def get_menu(probe_image: Image) -> Image:
    probe_image_gray = np.array(probe_image.convert("L"))
    # compute vertical edges
    sobel_x = cv2.Sobel(probe_image_gray, cv2.CV_64F, 1, 0, ksize=5)
    # compute horizontal edges
    sobel_y = cv2.Sobel(probe_image_gray, cv2.CV_64F, 0, 1, ksize=5)

    # average the vertical and horizontal edges over the image
    mean_sobel_x = np.mean(sobel_x, axis=0)
    mean_sobel_y = np.mean(sobel_y, axis=1)

    # get the absolute values
    mean_sobel_x = np.abs(mean_sobel_x)
    mean_sobel_y = np.abs(mean_sobel_y)

    # set a threshold for the edges to be within the image
    edge_threshold = 0.01
    # zero out the sobel values that are too close to the edge
    mean_sobel_x[:int(edge_threshold*len(mean_sobel_x))] = 0
    mean_sobel_x[int((1-edge_threshold)*len(mean_sobel_x)):] = 0
    mean_sobel_y[:int(edge_threshold*len(mean_sobel_y))] = 0
    mean_sobel_y[int((1-edge_threshold)*len(mean_sobel_y)):] = 0

    # perform peak finding on the mean values
    # pick a threshold that's the 90th percentile of the mean values
    threshold_sobel_x = np.percentile(mean_sobel_x, 90)
    threshold_sobel_y = np.percentile(mean_sobel_y, 90)
    # find peaks above the threshold
    peaks_x, _ = find_peaks(mean_sobel_x, height=threshold_sobel_x)
    peaks_y, _ = find_peaks(mean_sobel_y, height=threshold_sobel_y)

    # we only want the 2 largest peaks for the vertical edges to denote the width of the menu
    target_peaks_x = np.sort(mean_sobel_x[peaks_x])[-2:]
    target_peaks_x_coords = peaks_x[np.argsort(mean_sobel_x[peaks_x])[-2:]]

    # check if the max peak for horizontal edges is in the center of the image, if so, don't use it
    center_threshold = 0.1
    if np.abs(np.argmax(mean_sobel_y) - len(mean_sobel_y)/2) < center_threshold*len(mean_sobel_y):
        target_peaks_y = np.sort(mean_sobel_y[peaks_y])[-3:-1]
        target_peaks_y_coords = peaks_y[np.argsort(mean_sobel_y[peaks_y])[-3:-1]]
    else:
        target_peaks_y = np.sort(mean_sobel_y[peaks_y])[-2:]
        target_peaks_y_coords = peaks_y[np.argsort(mean_sobel_y[peaks_y])[-2:]]

    # crop the image using the target peaks
    # convert coordinates to integers
    # sort the coordinates in case they are not in the correct order
    target_peaks_x_coords = np.sort(target_peaks_x_coords)
    target_peaks_y_coords = np.sort(target_peaks_y_coords)
    target_peaks_x_coords = target_peaks_x_coords.astype(int)
    target_peaks_y_coords = target_peaks_y_coords.astype(int)
    menu_image = probe_image.crop((target_peaks_x_coords[0], target_peaks_y_coords[0], target_peaks_x_coords[1], target_peaks_y_coords[1]))

    # remove alpha channel if it exists
    if menu_image.mode == 'RGBA':
        menu_image = menu_image.convert('RGB')

    return menu_image


def split_menu(menu_image: Image, mode: str) -> list[Image]:
    # read in the template images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == 'Boss Specific':
        template_filepath = 'assets/boss_mode_row.png'
    else:
        template_filepath = 'assets/overall_mode_row.png'
    template_filepath = os.path.join(current_dir, '..', template_filepath)
    template = Image.open(template_filepath)

    # resize the template to match the cropped image's width and keep the aspect ratio
    new_width = menu_image.width
    new_height = int(template.height * new_width / template.width)
    resized_template = template.resize((new_width, new_height), Image.BICUBIC)
    resized_template = np.array(resized_template)
    np_menu_image = np.array(menu_image)

    # find the number of templates in the cropped image
    resized_template_gray = cv2.cvtColor(resized_template, cv2.COLOR_RGB2GRAY)
    menu_image_gray = cv2.cvtColor(np_menu_image, cv2.COLOR_RGB2GRAY)
    method = cv2.TM_CCOEFF_NORMED
    # match_result will be a 1D array of the match values because we matched widths
    match_result = cv2.matchTemplate(menu_image_gray, resized_template_gray, method)

    # perform peak finding on the match result to find of rows
    peaks, _ = find_peaks(match_result[:,0], height=np.median(match_result[:,0]), distance=3/4*new_height)

    cropped_rows = []
    for peak in peaks:
        top_left = (0, peak)
        bottom_right = (top_left[0] + new_width, top_left[1] + new_height)
        # crop the image using the target peaks
        cropped_row = menu_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
        cropped_rows.append(cropped_row)

    return cropped_rows
    
def get_portraits(input_image: Image) -> list[Image]:
    # resize the image to a higher resolution if needed
    # try to get 256x1024
    # compute scale factor from the original image
    resize_factor = 256 / input_image.height
    resized_image = input_image.resize((int(input_image.width * resize_factor), 256), 
        Image.BICUBIC)
    np_resized_image = np.array(resized_image)
    gray_resized_image = resized_image.convert('L')
    np_gray_resized_image = np.array(gray_resized_image)

    # binarize the image to make the watershed algorithm work better
    threshold = filters.threshold_otsu(np_gray_resized_image)
    binary = np_gray_resized_image > threshold
    binary = (binary).astype(np.uint8) * 255

    # find continuous region (low gradient - where less than N) --> markers
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(binary, disk(2))
    markers = gradient < 10
    markers = ndi.label(markers)[0]
    labels = watershed(gradient, markers)

    # remove region with the most common label - background
    c = Counter(labels.flatten())
    background_label = c.most_common(1)[0][0]

    # set any segments that are mostly blue to the background label
    blue = np.array([32, 129, 206]) # manually picked from sample image
    # get the mean color of each region
    # labels start at 1
    unique_labels = np.unique(labels)
    region_means = {i: np.mean(np_resized_image[labels == i], axis=0) for i in unique_labels}
    # compute the color distance
    color_distance = {i: np.linalg.norm(region_means[i] - blue) for i in unique_labels}
    # reject the region that is mostly blue
    postproc_labels_1 = labels.copy()
    for i in unique_labels:
        if color_distance[i] < 0.2:
            postproc_labels_1[postproc_labels_1 == i] = background_label

    # repeat the above for red instead of blue
    # red = np.array([163,43,39]) # manually picked from sample image
    # # get the mean color of each region
    # # labels start at 1
    # unique_labels = np.unique(labels)
    # region_means = {i: np.mean(np_resized_image[labels == i], axis=0) for i in unique_labels}
    # # compute the color distance
    # color_distance = {i: np.linalg.norm(region_means[i] - red) for i in unique_labels}
    # # reject the region that is mostly red
    # postproc_labels_1 = labels.copy()
    # for i in unique_labels:
    #     if color_distance[i] < 0.2:
    #         postproc_labels_1[postproc_labels_1 == i] = background_label

    # remove the background label
    postproc_labels_2 = postproc_labels_1.copy()
    postproc_labels_2[postproc_labels_2 == background_label] = 0
    # set all others to 1
    postproc_labels_2[postproc_labels_2 != 0] = 1

    # connected components on the binarized label image
    postproc_labels_3 = measure.label(postproc_labels_2, background=0)
    # get the region properties
    region_properties = regionprops(postproc_labels_3)

    # reject the small regions
    # get the area of each region
    minimum_x_length = 0.1 * resized_image.width
    minimum_y_length = 0.2 * resized_image.height
    minimum_area = minimum_x_length * minimum_y_length

    # remove regions that are too small
    postproc_labels_4 = postproc_labels_3.copy()
    for i in range(len(region_properties)):
        region_area_assuming_rect = (region_properties[i].bbox[2] - region_properties[i].bbox[0]) * \
                                (region_properties[i].bbox[3] - region_properties[i].bbox[1])
        if region_area_assuming_rect < minimum_area:
            # background is labeled as 0 b/c measure.label starts at 1 for the labels
            postproc_labels_4[postproc_labels_4 == region_properties[i].label] = 0

    # get the bounding box for each region by the min and max values of the x and y coordinates
    # don't care about the background
    region_properties_4 = regionprops(postproc_labels_4)
    bounding_boxes = [region.bbox for region in region_properties_4]
    # bounding box is (min_row, min_col, max_row, max_col)
    # crop the image to the bounding box
    # PIL crop is (left, upper, right, lower) so convert the bounding box order
    bounding_boxes = [(bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]) for bounding_box in bounding_boxes]
    cropped_images = [resized_image.crop(bounding_box) for bounding_box in bounding_boxes]

    return cropped_images