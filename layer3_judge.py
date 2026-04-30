# ===== LAYER 3: PREDICTIVE STRATEGIST (OPPORTUNITY UPDATE) =====
import numpy as np

class LLMJudge:
    def __init__(self):
        self.base_threshold = 0.51 # Lowered from 0.55 to catch more moves
        self.risk_per_trade = 0.02

    def run_shadow_simulations(self, current_price, df, hours_ahead=8, simulations=1000):
        """
        Simulates 1000 futures. We use 8 hours to bridge the 4-hour GitHub gap.
        """
        returns = df['close'].pct_change().dropna()
        if returns.empty:
            return 0.5, current_price, current_price * 0.98
            
        mu = returns.mean()
        sigma = returns.std()
        
        # Monte Carlo paths
        shocks = np.random.normal(mu, sigma, (simulations, hours_ahead))
        price_paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        final_prices = price_paths[:, -1]
        
        win_prob = np.sum(final_prices > current_price) / simulations
        expected_price = np.mean(final_prices)
        worst_case = np.percentile(final_prices, 5) 
        
        return float(win_prob), float(expected_price), float(worst_case)

    def evaluate(self, symbol, tech, ml, df):
        current_price = float(df.iloc[-1]['close'])
        returns = df['close'].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.015
        
        # 1. RUN SHADOW FORESIGHT
        shadow_win_prob, exp_price, shadow_risk = self.run_shadow_simulations(current_price, df)

        # 2. OPPORTUNITY LOGIC (Asymmetric Tuning)
        is_alt = any(alt in symbol for alt in ["SOL", "DOGE", "PEPE"])
        
        # Base confidence calculation
        tech_prob = (tech + 1) / 2
        confidence = (0.4 * shadow_win_prob) + (0.3 * ml) + (0.3 * tech_prob)

        # --- THE OPPORTUNITY INJECT ---
        if is_alt:
            # Alts get a 'Vibe Boost'—if they show even slight momentum, we lean in
            confidence *= 1.15 
            current_threshold = self.base_threshold - 0.04 # Much easier to trigger
            regime_type = "opportunity_hunt"
        else:
            # BTC remains the disciplined anchor
            current_threshold = self.base_threshold
            regime_type = "stable_growth"

        confidence = float(np.clip(confidence, 0, 1))

        # 3. DYNAMIC ACTIONS
        if confidence > current_threshold:
            action = 'BUY'
        elif confidence < (current_threshold - 0.10) or current_price < shadow_risk:
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            "action": action,
            "confidence": confidence,
            "threshold": current_threshold,
            "regime": regime_type,
            "shadow_win_prob": shadow_win_prob,
            "expected_future_price": exp_price,
            "shadow_risk_floor": shadow_risk
        }
