import requests
from PIL import Image
import os
from bs4 import BeautifulSoup
import time

webpage_url = 'https://dotgg.gg/nikke/characters'
http_response = requests.get(webpage_url)
parsed_html_content = BeautifulSoup(http_response.content, 'html.parser')

image_save_directory = "nikke_database"
image_url_substring = "/nikke/images/characters/"

for image_tag in parsed_html_content.find_all('img'):
    if image_url_substring in image_tag['src']:
        character_name = image_tag['alt']
        image_url = "https://dotgg.gg" + image_tag['src']
        image_response = requests.get(image_url)
        with open(image_save_directory + "/" + character_name, "wb") as image_file:
            image_file.write(image_response.content)
            try:
                downloaded_image = Image.open(image_save_directory + "/" + character_name)
                downloaded_image.verify()
            except Exception:
                image_file.close()
                os.remove(image_file.name)
        # rate limit to 10 requests per second at most
        time.sleep(0.1)