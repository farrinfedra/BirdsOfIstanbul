
from selenium import webdriver
from selenium.webdriver.common.by import By
import numpy as np

driver = webdriver.Chrome('/Users/farrinsofian/Documents/chromium/chromedriver')

#driver.get("https://ebird.org/region/TR")
driver.get("https://ebird.org/region/eu")



lists = []
scientific_name = []

def append_list(lis, scrapped):
    for i in range(len(scrapped)):
        lis.append(scrapped[i].text)


scientific_name = driver.find_elements(By.XPATH, "//a/span[@class='Heading-main']")

append_list(lists, scientific_name)


np.savetxt("ebird_europe.txt", np.asarray(lists), fmt="%s")



