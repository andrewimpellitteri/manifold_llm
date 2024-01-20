from manifoldpy import api
from manifoldpy.api import BinaryMarket

for market in api.get_all_markets(limit=300):

    if isinstance(market, BinaryMarket):
        print(market.question)
        print(market.closeTime)

    print("\n")