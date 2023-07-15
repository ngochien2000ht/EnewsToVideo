import os
from selenium.webdriver.common.by import By
import requests
import time


def link_img(string, wd):
	url = "https://www.google.co.in/search?q=" + str(string) + "&source=lnms&tbm=isch"
	wd.get(url)
	img_sr = wd.find_elements(By.CLASS_NAME, "Q4LuWd")
	img_sr[0].click()
	time.sleep(2)
	img_element = wd.find_element(By.CLASS_NAME, "r48jcc")
	image_url = img_element.get_attribute('src').split("?")[0]
	return image_url


def download_image(download_path, i, url):
	try:
		os.makedirs(download_path + str(i))
		if ".jpg" in url:
			filename = download_path + str(i) + "/my_image.jpg"
		if ".png" in url:
			filename = download_path + str(i) + "/my_image.png"
		if ".jpeg" in url:
			filename = download_path + str(i) + "/my_image.jpeg"
		response = requests.get(url)
		with open(filename, "wb") as f:
			f.write(response.content)
	except:
		os.rmdir(download_path + str(i))