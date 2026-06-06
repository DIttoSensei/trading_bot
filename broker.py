import requests
from typing import Dict, List, Optional

class Broker:
    """
    Interfaces directly with the Alpaca trade execution routing engines.
    """
    def __init__(self, api_key: str, secret_key: str):
        # Using paper trading endpoints for risk mitigation
        self.base_url = "https://paper-api.alpaca.markets/v2"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json"
        }

    def get_account(self):
        res = requests.get(f"{self.base_url}/account", headers=self.headers)
        if res.status_code == 200:
            data = res.json()
            class Account:
                def __init__(self, d):
                    self.equity = d.get("equity", 100000.0)
                    self.buying_power = d.get("buying_power", 200000.0)
            return Account(data)
        raise ValueError(f"Alpaca API connection failure: {res.text}")

    def get_position_info(self, symbol: str) -> dict:
        clean_symbol = symbol.replace("/", "")
        res = requests.get(f"{self.base_url}/positions/{clean_symbol}", headers=self.headers)
        if res.status_code == 200:
            return res.json()
        return {"qty": 0.0, "avg_entry_price": 0.0}

    def get_all_positions(self) -> List:
        res = requests.get(f"{self.base_url}/positions", headers=self.headers)
        if res.status_code == 200:
            raw_list = res.json()
            class PositionWrapper:
                def __init__(self, d):
                    self.qty = float(d.get("qty", 0.0))
                    self.current_price = float(d.get("current_price", 0.0))
                    self.symbol = d.get("symbol")
            return [PositionWrapper(p) for p in raw_list]
        return []

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str = "market"):
        clean_symbol = symbol.replace("/", "")
        payload = {
            "symbol": clean_symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "gtc"
        }
        res = requests.post(f"{self.base_url}/orders", json=payload, headers=self.headers)
        return res.json() if res.status_code == 200 else None