
# Description
The user supplies an in-game screenshot of the "Union Log" for the overall raid or on a per-boss basis, then the application extracts information such as damage dealt, commander name, and boss name.

"Overall" Mode:

https://github.com/SiftingSands/union_raid_log_extraction/assets/43226539/af790b2b-1712-4a85-8c5f-e9eca94a3e7d

"Boss Specific" Mode:

https://github.com/SiftingSands/union_raid_log_extraction/assets/43226539/d10a5bc1-1e4b-4cf9-b1eb-2251213cef95

# Background
- The relevant parts of the image containing a "hit" on the boss are extracted using "classical" image processing techniques such as Sobel edge detection, watershed segmentation, and template matching with the two `*.png` files in the  `assets` folder.
    - There are hard-coded parameters that have only been tested on 1440p in-game screenshots
- OCR leverages the mmocr library - https://github.com/open-mmlab/mmocr
    - Text detection and recognition use DBNet++ and ABINet models respectively
- Two modes selectable in the dashboard:
    - "Overall" mode: screenshot similar to the one below is used and the "commander_names, commander_damages, boss_names, boss_levels" are returned in a table that can be downloaded as a CSV
    - "Boss Specific" mode: screenshot similar to the one below is used and the "commander_names, commander_damages, team_composition, boss_levels, unit_levels" are returned in a table that can be downloaded as a CSV
- "Boss Specific" mode relies on "assets/nikke_images.pkl" which were generated using the 'nikke_puller.py' script
    - Simply run as `python utils/nikke_puller.py`. Filepath to save the results are hard-coded in the script.
    - The script will pull the images from a website containing portraits of all the units in the game, clean up the transparent background, and save the images as a pickle file.

# Limitations

1. Does NOT work on screenshots of the "Union Log" after the union raid ends. They are a white background that messes up the image processing pipeline that was designed for the dark background.
2. OCD may detect two closely spaced numbers for a commander's damage. OCR then occaisonally has failures with duplicate numbers, so the reported damage is too high due to an extra digit. For example, `123,123` vs `1,231,233`.
3. The number of portraits detected in the "Boss Specific" mode expects 6; 1 for the boss and 5 for the team composition. If the number of portraits detected is not 6, no team composition will be returned in the table. This is an observed occaisonal failure.
4. The matching algorithm is a simple template matching algorithm that uses the "assets/nikke_images.pkl" file to find the best match for each unit in the screenshot ("Boss Specific" mode). *However, matching accuracy is not great and the user may have to manually correct the results.* For example, Mary is often matched with the wrong unit. I suspect this is because additional information such as unit level and core level is overlaid on the in-game portraits.
    - I tried using feature embeddings from small CNNs such as VGG-16 and ResNet-18, but they performed worse than the template matching algorithm.

# Installation and Startup

## Run with Streamlit Cloud
1. Go to https://unionraidlogextraction-b7gosme4rcygjmhibee5de.streamlit.app/

OR

## Run with Python
0. Install Poetry https://python-poetry.org/docs/
1. Install all Python dependencies in your Poetry environment after navigating to this directory
    - `poetry install`
2. Run the dashboard
    - `streamlit run src/app.py`
    - OR use a different port other than the default 8501
        - `streamlit run src/app.py --server.port <port>`
3. Open up the dashboard in your browser if it doesn't open automatically
    - `http://localhost:<port>`

# Usage

1. Upload a screenshot of the "Union Log" on the right side of the dashboard

    Take in-game screenshots from these locations:
   
    "Overall"
   
   ![image](https://github.com/SiftingSands/union_raid_log_extraction/assets/43226539/80bff614-5d7c-4af6-8d14-dd4b9f9df78f)

   "Boss Specific"
   
   ![image(1)](https://github.com/SiftingSands/union_raid_log_extraction/assets/43226539/c9db15d6-e275-4c7e-a385-637e67cde6d2)


3. Select the mode you want to run
    - See `assets/overall_example.png` and `assets/boss_specific_example.png` for sample inputs for each mode
4. Click the "Display Intermediate Images" checkbox if you want to see the intermediate images used in the extraction process
5. Click the "Run" button
6. The results will be displayed in the dashboard. The free tier of streamlit cloud is CPU only, so results for a single image may take up to 1 minute to be computed.
7. Upload another image and repeat if desired

# Misc
- `utils/visualize_raid_results.py` can be ran to create Plotly graphs of the overall union raid results. Result samples from season 7 are included in `assets/*.csv`
