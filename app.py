import finnhub

# Setup client
api_key = 'd342clhr01qqt8sn7pdgd342clhr01qqt8sn7pe0'
finnhub_client = finnhub.Client(api_key=api_key)

# Example: Get quote for Apple
quote = finnhub_client.quote('AAPL')
print(quote)
