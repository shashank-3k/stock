import requests
import logging

api_keys_eod = ["64804202d365e9.21480491",
            "6481b6b56bdf05.37294726",
            "6481b11d245909.63698937",
            "64116aaa61d427.97113357",
            "6486fa945b5c66.26937899",
            "6486fadb900500.31198539",
            "6486fe289acac0.61254586",
            "648afe3eb27ef0.53167560",
            "648afeddab03b1.26462561",
            "648aff57ca5d45.40621607"]

url = "https://eodhistoricaldata.com/api/" #sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token={}".format(stock,key)

def make_api_request_eod(stock):
    for key in api_keys_eod:
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



api_keys_av = [
                "VOZBT17O7CX766DW",
                "A8K0KZWW0WYV2SHC",
                "2Z4Y72ZC82AE904V",
                "7MF51BVY1PHGWMK8",
                "L82N6LJGLR5DP8L9",
                "96TJZH2I0UU7ONLD",
                "QXHAR6EFSLX8XF19",
                "T3QQ6KR5YFUZLZQ3",
                "TEOSZZY21UXG4N2C",
                "62M8CQZHYL4KH8JH"]

url_av = "https://www.alphavantage.co/query?"  #function=BALANCE_SHEET&symbol={ticker_symbol}&apikey={api_key}

def make_api_request_av(stock, func):
    for key in api_keys_av:
        try:
            url_report = url_av + "function={}&symbol={}&apikey={}".format(func,stock,key)
            response_report = requests.get(url_report)
            response_report.raise_for_status()  # Raise an exception for non-2xx status codes

            return 1, response_report.json()
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error occurred for API key: {key}")
            logging.error(f"Error details: {str(e)}")
            continue  # Move on to the next key

    logging.error("All AlphaVantage API keys have encountered errors or reached the rate limit.")
    return 0, None
