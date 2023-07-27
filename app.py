import streamlit as st
from segmentation import get_menu, split_menu, get_portraits
from PIL import Image, ImageDraw
from mmlab_ocr import run_ocr, parse_ocr_overall_results, parse_ocr_boss_specific_results
from matcher import match_portrait
import pandas as pd
import numpy as np
import re

def mode_0(image):
    # split the menu
    menu_images = split_menu(menu, mode='Boss Specific')
    if display_intermediate_images:
        # display the menu items
        for i, menu_image in enumerate(menu_images):
            st.image(menu_image, caption=f"Menu Item {i}", use_column_width=True)

    # perform OCR on the menu items
    commander_names = []
    commander_damages = []
    team_composition = []
    unit_levels = []
    boss_levels = []
    for i, menu_image in enumerate(menu_images):
        bgr_menu_image = np.array(menu_image)
        bgr_menu_image = bgr_menu_image[:, :, ::-1].copy()
        # get the height and width of the image
        width, height = menu_image.size
        
        ocr_result = run_ocr(bgr_menu_image)
        rec_texts = ocr_result['predictions'][0]['rec_texts']
        det_polygons = ocr_result['predictions'][0]['det_polygons']
        commander_damage, commander_name, unit_level, boss_level = parse_ocr_boss_specific_results(rec_texts, det_polygons, width, height)

        if display_intermediate_images:
            # draw the bounding boxes but don't modify the original image
            menu_image_display = menu_image.copy()
            draw = ImageDraw.Draw(menu_image_display)
            for polygon in det_polygons:
                draw.polygon(polygon, outline="red", width=3)
            # display the image with bounding boxes
            st.markdown(f"Menu Item {i} with Text Detections")
            st.image(menu_image_display, caption=f"Menu Item {i}", use_column_width=True)

        portraits = get_portraits(menu_image)
        if len(portraits) != 6:
            st.markdown(f"For Menu Item {i+1}, found {len(portraits)} portraits instead of 6. Only reporting OCR results.")
            portrait_error_flag = True
        else:
            portrait_error_flag = False

        # TODO : currently hard coded to skip the boss portrait, probably not an actual needed feature
        skip_first_portrait = True
        if portrait_error_flag is False:
            portrait_IDs = []
            template_images_folderpath = 'nikke_database'
            for i, portrait in enumerate(portraits):
                if skip_first_portrait and i == 0:
                    continue
                portrait_ID = match_portrait(portrait, template_images_folderpath)
                portrait_IDs.append(portrait_ID)

        if display_intermediate_images:
            # display the portraits
            st.markdown(f"Menu Item {i} Portraits")
            for j, portrait in enumerate(portraits):
                if skip_first_portrait and j == 0:
                    continue
                st.image(portrait, caption=f"Portrait {j}", use_column_width=True)
                if portrait_error_flag is False:
                    if skip_first_portrait:
                        # off by 1 because we skipped the first portrait
                        st.markdown(f"**Portrait {j} ID: {portrait_IDs[j-1]}**")
                    else:
                        st.markdown(f"**Portrait {j} ID: {portrait_IDs[j]}**")
                else:
                    st.markdown(f"**Portrait {j} ID: N/A**")

        # collect the results
        commander_names.append(commander_name)
        commander_damages.append(commander_damage)
        unit_levels.append(unit_level)
        boss_levels.append(boss_level)
        if portrait_error_flag is False:
            team_composition.append(portrait_IDs)
        else:
            team_composition.append('N/A')

    return commander_names, commander_damages, team_composition, boss_levels, unit_levels

def mode_1(image):
    # split the menu
    mode = 'Overall'
    menu_images = split_menu(menu, mode)
    if display_intermediate_images:
        # display the menu items
        for i, menu_image in enumerate(menu_images):
            st.image(menu_image, caption=f"Menu Item {i}", use_column_width=True)
    
    # perform OCR on the menu items
    commander_names = []
    commander_damages = []
    boss_names = []
    boss_levels = []
    for i, menu_image in enumerate(menu_images):
        bgr_menu_image = np.array(menu_image)
        bgr_menu_image = bgr_menu_image[:, :, ::-1].copy()
            # get the height and width of the image
        width, height = menu_image.size
        
        ocr_result = run_ocr(bgr_menu_image)
        rec_texts = ocr_result['predictions'][0]['rec_texts']
        det_polygons = ocr_result['predictions'][0]['det_polygons']
        commander_damage, commander_name, boss_name, boss_level = parse_ocr_overall_results(rec_texts, det_polygons, width, height)

        if display_intermediate_images:
            # draw the bounding boxes but don't modify the original image
            menu_image_display = menu_image.copy()
            draw = ImageDraw.Draw(menu_image_display)
            for polygon in det_polygons:
                draw.polygon(polygon, outline="red", width=3)
            # display the image with bounding boxes
            st.markdown(f"Menu Item {i} with Text Detections")
            st.image(menu_image_display, caption=f"Menu Item {i}", use_column_width=True)
        
        # collect the results
        commander_names.append(commander_name)
        commander_damages.append(commander_damage)
        boss_names.append(boss_name)
        boss_levels.append(boss_level)

    return commander_names, commander_damages, boss_names, boss_levels


if __name__ == "__main__":
    # Title
    st.title("Image Processing App")

    # Sidebar to load an image
    st.sidebar.title("Load Image")
    # any image type can be uploaded
    input_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if input_image is not None:
        st.sidebar.markdown("Input image loaded successfully")
    else:
        st.sidebar.markdown("Please upload an image")

    # Create a run button on the main screen if the image is loaded
    if input_image is not None:
        run = st.button("Run")

    # Toggle button for displaying intermediate images
    display_intermediate_images = st.sidebar.checkbox("Display Intermediate Images", value=False)

    # Toogle button for mode selection
    mode = st.sidebar.radio("Mode", ["Overall", "Boss Specific"])

    # Main function
    if input_image is not None and run:
        # load the image
        image = Image.open(input_image)
        # display the image
        st.image(image, caption="Input Image", use_column_width=True)

        menu = get_menu(image)
        if display_intermediate_images:
            # display the menu
            st.image(menu, caption="Full Menu", use_column_width=True)

        # get the menu
        if mode == "Boss Specific":
            commander_names, commander_damage, team_composition, boss_levels, unit_levels = mode_0(image)
            results = pd.DataFrame({"Commander Name": commander_names, "Commander Damage": commander_damage, 
                                    "Team Composition": team_composition, "Boss Level": boss_levels, "Unit Levels": unit_levels})
        else:
            commander_names, commander_damage, boss_name, boss_level = mode_1(image)
            results = pd.DataFrame({"Commander Name": commander_names, "Commander Damage": commander_damage, 
                                    "Boss Name": boss_name, "Boss Level": boss_level})

        # display the results in a table
        st.markdown("## Tabulated Results")
        results['Commander Damage'] = results['Commander Damage'].fillna(0).astype(int)
        results['Boss Level'] = results['Boss Level'].fillna(0).astype(int)
        st.dataframe(results)

        # download the dataframe as a csv at the click of a button
        csv = results.to_csv(index=False)
        # https://github.com/streamlit/streamlit/issues/4382
        st.markdown("Unfortunately, clicking the download button will clear main screen. \
            Please take a screenshot of the results before clicking the button if needed.")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )
