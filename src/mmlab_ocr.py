from mmocr_inference_mod import MMOCRInferencer_merged_dets
import numpy as np
import re

def run_ocr(input_image: np.ndarray, intersection_threshold: float = 1e-2, min_area: int = 250,
            det_score_threshold: float = 0.4):
    engine = MMOCRInferencer_merged_dets(det='dbnetpp', rec='ABINet_Vision', intersection_threshold=intersection_threshold,
        min_area=min_area, det_score_threshold=det_score_threshold)
    result = engine(input_image, return_vis=False)
    return result

def parse_ocr_overall_results(rec_texts: list, det_polygons: list, width: int, height: int):
    """ Parse the OCR results from MMOCR to find commander name, damage done, boss name, and boss level.
    """
    # Boss level starts with "lv" and ends in a 1-2 digit number
    # remove the text and polygon after the first match
    boss_level = None
    for i, text in enumerate(rec_texts):
        # only consider polygons in the upper left quadrant of the image
        if det_polygons[i][0] > 0.5 * width or det_polygons[i][1] > 0.5 * height:
            continue
        # use lower case to make it case insensitive
        if text.lower().startswith('lv'):
            # split by 1st non-digit character
            boss_level = re.split('\D+', text)[1]
            # if the above doesn't return a number, the number might have been switched to a letter that looks like a number
            if not boss_level.isdigit():
                # remove the 'lv' prefix
                boss_level = text.replace('lv', '')
                # try to convert the letter to a number
                boss_level = boss_level.replace('o', '0')
                boss_level = boss_level.replace('O', '0')
                boss_level = boss_level.replace('l', '1')
                boss_level = boss_level.replace('I', '1')
                boss_level = boss_level.replace('i', '1')
                boss_level = boss_level.replace('z', '2')
                boss_level = boss_level.replace('Z', '2')
                boss_level = boss_level.replace('s', '5')
                boss_level = boss_level.replace('S', '5')
                boss_level = boss_level.replace('b', '8')
                boss_level = boss_level.replace('B', '8')
                boss_level = boss_level.replace('g', '9')
                boss_level = boss_level.replace('G', '9')
                boss_level = boss_level.replace('q', '9')
                boss_level = boss_level.replace('Q', '9')
                # if the above still doesn't return a number, then we don't have a boss level
                if not boss_level.isdigit():
                    boss_level = None
                    continue

            index_to_remove = i
            break
    
    # make a copy of the lists so we don't modify the original
    # otherwise, screwy things happen in the dashboard
    rec_texts_copy = rec_texts.copy()
    det_polygons_copy = det_polygons.copy()
    if boss_level is not None:
        rec_texts_copy.pop(index_to_remove)
        det_polygons_copy.pop(index_to_remove)
    
    # Damage done is a big number, typically greater than 1 million
    commander_damage_candidates = []
    for i, text in enumerate(rec_texts_copy):
        # check if the text is a number after removing commas
        if text.replace(',', '').isdigit():
            # corresponding polygon should be in the upper right quadrant of the image
            if det_polygons_copy[i][0] > 0.5 * width and det_polygons_copy[i][1] < 0.5 * height:
                commander_damage_candidates.append(text)
    # merge the numbers into a single string from left to right based on their polygon coordinates
    commander_damage_candidates = sorted(commander_damage_candidates, key=lambda x: det_polygons_copy[rec_texts_copy.index(x)][0])
    # if there are multiple numbers, then they sometimes have a duplicate number at the end of one and the start of the next
    # remove the duplicate number
    for i in range(len(commander_damage_candidates) - 1):
        if commander_damage_candidates[i][-1] == commander_damage_candidates[i+1][0]:
            commander_damage_candidates[i] = commander_damage_candidates[i][:-1]
    commander_damage = ''.join(commander_damage_candidates)
    # # remove any numbers that aren't 3 digits long in-between commas
    # commander_damage = re.sub(r'(?<=,)\d{1,2}(?=,)', '', commander_damage)
    # # remove any commas that aren't between 3 digit numbers
    # commander_damage = re.sub(r'(?<=\d),(?=\d)', '', commander_damage)

    # Commander name should be the longest string of text in the upper left quadrant of the image
    commander_name_candidates = []
    for i, text in enumerate(rec_texts_copy):
        if det_polygons_copy[i][0] < 0.5 * width and det_polygons_copy[i][1] < 0.5 * height:
            commander_name_candidates.append(text)
    # commander names can't have spaces so we only need the longest string
    if len(commander_name_candidates) > 0:
        commander_name = max(commander_name_candidates, key=len)
    else:
        commander_name = None

    # Boss name should be the longest string of text in the lower left quadrant of the image
    boss_name_candidates = []
    for i, text in enumerate(rec_texts_copy):
        if det_polygons_copy[i][0] < 0.5 * width and det_polygons_copy[i][1] > 0.5 * height:
            boss_name_candidates.append(text)
    # merge the names into a single string from left to right based on their polygon coordinates
    boss_name_candidates = sorted(boss_name_candidates, key=lambda x: det_polygons_copy[rec_texts_copy.index(x)][0])
    # add a space between each name
    boss_name = ' '.join(boss_name_candidates)

    return commander_damage, commander_name, boss_name, boss_level

def parse_ocr_boss_specific_results(rec_texts: list, det_polygons: list, width: int, height: int):
    """ Parse the OCR results from MMOCR to find commander name, unit level, damage done, and boss level.
    """
    # Boss level starts with "lv" and ends in a 1-2 digit number
    # remove the text and polygon after the first match
    boss_level = None
    for i, text in enumerate(rec_texts):
        # only consider polygons in the upper left quadrant of the image
        if det_polygons[i][0] > 0.5 * width or det_polygons[i][1] > 0.5 * height:
            continue
        # use lower case to make it case insensitive
        if text.lower().startswith('lv'):
            # split by 1st non-digit character
            boss_level = re.split('\D+', text)[1]
            # if the above doesn't return a number, the number might have been switched to a letter that looks like a number
            if not boss_level.isdigit():
                # remove the 'lv' prefix
                boss_level = text.replace('lv', '')
                # try to convert the letter to a number
                boss_level = boss_level.replace('o', '0')
                boss_level = boss_level.replace('O', '0')
                boss_level = boss_level.replace('l', '1')
                boss_level = boss_level.replace('I', '1')
                boss_level = boss_level.replace('i', '1')
                boss_level = boss_level.replace('z', '2')
                boss_level = boss_level.replace('Z', '2')
                boss_level = boss_level.replace('s', '5')
                boss_level = boss_level.replace('S', '5')
                boss_level = boss_level.replace('b', '8')
                boss_level = boss_level.replace('B', '8')
                boss_level = boss_level.replace('g', '9')
                boss_level = boss_level.replace('G', '9')
                boss_level = boss_level.replace('q', '9')
                boss_level = boss_level.replace('Q', '9')
                # if the above still doesn't return a number, then we don't have a boss level
                if not boss_level.isdigit():
                    boss_level = None
                    continue

            index_to_remove = i
            break
    
    # make a copy of the lists so we don't modify the original
    # otherwise, screwy things happen in the dashboard
    rec_texts_copy = rec_texts.copy()
    det_polygons_copy = det_polygons.copy()
    if boss_level is not None:
        rec_texts_copy.pop(index_to_remove)
        det_polygons_copy.pop(index_to_remove)

    # commander name is the longest string of characters in the upper third and left half of the image
    commander_name_candidates = []
    for i, text in enumerate(rec_texts_copy):
        # only consider polygons in the upper left quadrant of the image
        if det_polygons_copy[i][0] > 0.5 * width or det_polygons_copy[i][1] > 0.5 * height:
            continue
        # only consider polygons in the upper third of the image
        if det_polygons_copy[i][1] > 0.33 * height:
            continue
        commander_name_candidates.append(text)
    # commander names can't have spaces so we only need the longest string
    if len(commander_name_candidates) > 0:
        commander_name = max(commander_name_candidates, key=len)
    else:
        commander_name = None
    
    # Damage done is in the middle third and left half of the image
    commander_damage_candidates = []
    for i, text in enumerate(rec_texts_copy):
        # only consider polygons in the upper left quadrant of the image
        if det_polygons_copy[i][0] > 0.5 * width or det_polygons_copy[i][1] > 0.5 * height:
            continue
        # only consider polygons in the middle third of the image
        if det_polygons_copy[i][1] < 0.33 * height or det_polygons_copy[i][1] > 0.66 * height:
            continue
        commander_damage_candidates.append(text)
    # merge the numbers into a single string from left to right based on their polygon coordinates
    commander_damage_candidates = sorted(commander_damage_candidates, key=lambda x: det_polygons_copy[rec_texts_copy.index(x)][0])
    # if there are multiple numbers, then they sometimes have a duplicate number at the end of one and the start of the next
    # remove the duplicate number
    for i in range(len(commander_damage_candidates) - 1):
        if commander_damage_candidates[i][-1] == commander_damage_candidates[i+1][0]:
            commander_damage_candidates[i] = commander_damage_candidates[i][:-1]
    commander_damage = ''.join(commander_damage_candidates)

    # commander level is in the lower third of the image
    unit_level_candidates = []
    for i, text in enumerate(rec_texts_copy):
        # only consider polygons in the lower third of the image
        if det_polygons_copy[i][1] < 0.66 * height:
            continue
        unit_level_candidates.append(text)
    # find the most common level
    if len(unit_level_candidates) > 0:
        unit_level = max(set(unit_level_candidates), key=unit_level_candidates.count)
    else:
        unit_level = None

    return commander_damage, commander_name, unit_level, boss_level