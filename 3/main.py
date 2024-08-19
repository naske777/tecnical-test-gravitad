import json
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3', 'Accept-Language': 'en-US, en;q=0.5'})

    URL = "https://www.tuambia.com/?srsltid=AfmBOoon-iYKoVKyo99b997Y1RuOajjRBp05HI_PGKyTshCmOHu8XPvj"

    webpage = requests.get(URL, headers=HEADERS)

    # Create the Soup object containing all the data
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Find all product containers
    products = soup.find_all("div", class_="MuiCardContent-root flex-grow pb-1 py-1 px-2 md:py-2 md:px-4 css-1qw96cp")

    product_list = []

    # Loop to extract product details
    for product in products:
        product_details = {}
        try:
            product_details['title'] = product.find("h3", class_="MuiTypography-root MuiTypography-h3 mt-2 md:mt-0 css-12r7xb8").text.strip()
        except AttributeError:
            product_details['title'] = ""
        
        try:
            product_details['price'] = product.find("p", class_="MuiTypography-root MuiTypography-body1 css-4keu8q").text.strip()
            product_details['price'] = float(product_details['price'].replace("\xa0USD", "").replace(",", "."))
            
        except AttributeError:
            product_details['price'] = ""
        
        product_list.append(product_details)

    df = pd.DataFrame(product_list)
    df['title'] = df['title'].replace('', np.nan)
    df = df.dropna(subset=['title'])

    df.to_json('tuambia_products.json', orient='records', indent=4, date_format='iso', force_ascii=False)