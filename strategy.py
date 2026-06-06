import numpy as np
import pandas as pd
from typing import Dict, Optional
from layer1_technical import technical_bot

class TradingStrategy:
    """
    Evaluates raw historical indicators to generate structural 
    entry and exit confirmations.
    """
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, has_position: bool) -> Optional[Dict]:
        if len(df) < 30:
            return None
            
        tech_val = technical_bot(df)
        close = df['close'].iloc[-1]
        
        if not has_position:
            # Entry Signal Parameters
            if tech_val > 0.65:
                return {"action": "BUY", "confidence": tech_val, "reason": "structural_trigger"}
        else:
            # Exit Signal Parameters
            if tech_val < 0.35:
                return {"action": "SELL", "confidence": tech_val, "reason": "momentum_exhaustion"}
                
        return {"action": "HOLD", "confidence": tech_val, "reason": "continuation"}