import requests
import logging

api_keys = ["64804202d365e9.21480491",
            "6481b6b56bdf05.37294726",
            "6481b11d245909.63698937",
            "64116aaa61d427.97113357",
            "6486fa945b5c66.26937899",
            "6486fadb900500.31198539",
            "6486fe289acac0.61254586"]

url = "https://eodhistoricaldata.com/api/" #sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token={}".format(stock,key)

def make_api_request(stock):
    for key in api_keys:
        try:
            url_sentiment = url+ "sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token={}".format(stock,key)
            response_sentiment = requests.get(url_sentiment)
            response_sentiment.raise_for_status()  # Raise an exception for non-2xx status codes

            url_tweet = url+ "/tweets-sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token={}".format(stock,key)
            response_tweet = requests.get(url_tweet)
            response_tweet.raise_for_status()  # Raise an exception for non-2xx status codes
            return 1, response_sentiment.json(), response_tweet.json()
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error occurred for API key: {key}")
            logging.error(f"Error details: {str(e)}")
            continue  # Move on to the next key

    logging.error("All API keys have encountered errors or reached the rate limit.")
    return 0, None, None

