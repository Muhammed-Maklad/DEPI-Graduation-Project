import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time


def scrape_amazon_reviews(reviews_url, len_page=4):
    # Header to set the requests as a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

    # Function to get HTML pages from Amazon review pages
    def reviewsHtml(url, len_page):
        soups = []
        for page_no in range(1, len_page + 1):
            params = {
                'ie': 'UTF8',
                'reviewerType': 'all_reviews',
                'filterByStar': 'critical',
                'pageNumber': page_no,
            }
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                soup = BeautifulSoup(response.text, 'html.parser')
                soups.append(soup)
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch page {page_no}: {e}")
            # Add delay to avoid being blocked by Amazon
            time.sleep(2)
        return soups

    # Function to get reviews name, description, date, stars, title from HTML
    def getReviews(html_data):
        data_dicts = []
        boxes = html_data.find_all('div', {'data-hook': 'review'})

        for box in boxes:
            try:
                name = box.find(class_='a-profile-name').text.strip()
            except Exception:
                name = 'N/A'

            try:
                stars = box.find(
                    'i', {'data-hook': 'review-star-rating'}).text.strip().split(' out')[0]
            except Exception:
                try:
                    stars = box.find(
                        'span', {'class': 'a-icon-alt'}).text.strip().split(' out')[0]
                except Exception:
                    stars = 'N/A'

            try:
                title = box.find(
                    'a', {'data-hook': 'review-title'}).text.strip()
            except Exception:
                title = 'N/A'

            try:
                datetime_str = box.find(
                    'span', {'data-hook': 'review-date'}).text.strip().split(' on ')[-1]
                date = datetime.strptime(
                    datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
            except Exception:
                date = 'N/A'

            try:
                description = get_full_review_if_truncated(box)
            except Exception:
                description = 'N/A'

            data_dict = {
                'Name': name,
                'Stars': stars,
                'Title': title,
                'Date': date,
                'Description': description
            }
            data_dicts.append(data_dict)

        return data_dicts

    # Function to get the full review
    def get_full_review_if_truncated(review_element):
        # Get the main review text
        body = review_element.find(
            'span', {'data-hook': 'review-body'}).text.strip()

        # Check for "read more" link
        read_more_link = review_element.find(
            'a', {'data-hook': 'review-read-more-link'})
        if read_more_link:
            try:
                # Amazon links are relative, so we need to construct the full URL
                full_review_url = "https://www.amazon.com" + \
                    read_more_link['href']
                full_review_response = requests.get(
                    full_review_url, headers=headers)
                full_review_response.raise_for_status()  # Raise an HTTPError if bad response
                full_review_soup = BeautifulSoup(
                    full_review_response.content, 'html.parser')
                # Extract the full review text
                full_review_text = full_review_soup.find(
                    'span', {'data-hook': 'review-body'}).text.strip()
                return full_review_text
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch full review: {e}")

        return body

    # Function to classify sentiment based on star rating
    def classify_sentiment(star):
        try:
            star = float(star)  # Convert the star rating to a float
            if star < 3:
                return 'Negative'
            elif star == 3:
                return 'Neutral'
            else:
                return 'Positive'
        except ValueError:
            return 'Unknown'  # In case there's a non-numeric value

    # Grab all HTML
    html_datas = reviewsHtml(reviews_url, len_page)

    # Empty List to hold all reviews data
    reviews = []

    # Iterate all HTML pages
    for html_data in html_datas:
        review = getReviews(html_data)
        reviews += review

    # Create a DataFrame with reviews Data
    df_reviews = pd.DataFrame(reviews)

    df_reviews['Customer Review'] = df_reviews['Description'].str[:-10]
    df_reviews['Sentiment'] = df_reviews['Stars'].apply(classify_sentiment)
    df_reviews = df_reviews[['Sentiment', 'Customer Review']]
    # Save to CSV
    csv_filename = 'reviews.csv'
    df_reviews.to_csv(csv_filename, index=False)
    return csv_filename
