import json
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

# Function to extract the product title
def get_title(product):
    try:
        title = product.find("h3", class_="MuiTypography-root MuiTypography-h3 mt-2 md:mt-0 css-12r7xb8").text.strip()
    except AttributeError:
        title = ""
    return title

# Function to extract the product price
def get_price(product):
    try:
        price = product.find("p", class_="MuiTypography-root MuiTypography-body1 css-4keu8q").text.strip()
    except AttributeError:
        price = ""
    return price

if __name__ == '__main__':
    # Add your user agent
    HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3', 'Accept-Language': 'en-US, en;q=0.5'})

    # The URL of the webpage
    URL = "https://www.tuambia.com/?srsltid=AfmBOoon-iYKoVKyo99b997Y1RuOajjRBp05HI_PGKyTshCmOHu8XPvj"

    # Make the HTTP request
    webpage = requests.get(URL, headers=HEADERS)

    # Create the Soup object containing all the data
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Find all product containers
    products = soup.find_all("div", class_="MuiCardContent-root flex-grow pb-1 py-1 px-2 md:py-2 md:px-4 css-1qw96cp")

    # List to store product details
    product_list = []

    # Loop to extract product details
    for product in products:
        product_details = {}
        product_details['title'] = get_title(product)
        product_details['price'] = get_price(product)
        product_list.append(product_details)

    # Create a DataFrame with the extracted data
    df = pd.DataFrame(product_list)
    df['title'] = df['title'].replace('', np.nan)
    df = df.dropna(subset=['title'])

    # Save the data to a JSON file
    df.to_json("tuambia_products.json", orient='records', force_ascii=False)