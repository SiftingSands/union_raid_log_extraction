import requests
from PIL import Image
import os
from bs4 import BeautifulSoup
import time
import numpy as np
from io import BytesIO
import pickle

webpage_url = 'https://dotgg.gg/nikke/characters'
http_response = requests.get(webpage_url)
parsed_html_content = BeautifulSoup(http_response.content, 'html.parser')
image_url_substring = "/nikke/images/characters/"

images = {}
for image_tag in parsed_html_content.find_all('img'):
    if image_url_substring in image_tag['src']:
        character_name = image_tag['alt']
        image_url = "https://dotgg.gg" + image_tag['src']
        try:
            image_response = requests.get(image_url)
            # append the image data to the list of images
            image = Image.open(BytesIO(image_response.content)).convert('RGBA')
            # create a white background otherwise direct conversion to RGB has artifacts
            background = Image.new('RGBA', image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image)
            images[character_name] = np.array(alpha_composite)
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching image for {character_name}: {e}")
            continue
        # rate limit to 10 requests per second at most
        time.sleep(0.1)

# write the images to a pickle file
with open('./assets/nikke_images.pkl', 'wb') as f:
    pickle.dump(images, f)