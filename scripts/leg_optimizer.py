"""
Multi-Leg Strategy Optimizer

Finds optimal combinations of options legs for various strategies:
- Vertical spreads (credit/debit)
- Iron condors
- Butterflies
- Calendars

Optimizes for: max POP, max EV, min risk, delta neutrality
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)

from options_math import (
    BlackScholes, ProbabilityCalculator, Greeks,
    fits_account_constraints, optimal_spread_width,
    DEFAULT_ACCOUNT_TOTAL, ACCOUNT_TOTAL, MAX_RISK_PER_TRADE, AVAILABLE_CAPITAL
)
from chain_analyzer import OptionChain, ChainAnalyzer


@dataclass
class TradeLeg:
    """Single option leg in a multi-leg strategy"""
    strike: float
    expiration: str
    dte: int
    premium: float  # Per share
    option_type: str  # 'call' or 'put'
    action: str  # 'buy' or 'sell'
    quantity: int = 1
    greeks: Optional[Greeks] = None
    
    @property
    def net_premium(self) -> float:
        """Net premium for this leg (positive = credit, negative = debit)"""
        if self.action == 'sell':
            return self.premium * self.quantity
        else:
            return -self.premium * self.quantity


@dataclass
class MultiLegStrategy:
    """Complete multi-leg options strategy"""
    ticker: str
    strategy_type: str  # 'vertical_credit', 'vertical_debit', 'iron_condor', etc.
    underlying_price: float
    legs: List[TradeLeg] = field(default_factory=list)
    
    # Trade metrics
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakevens: List[float] = field(default_factory=list)
    pop: float = 0.0  # Probability of Profit
    expected_value: float = 0.0
    risk_adjusted_return: float = 0.0
    
    # Greeks
    total_greeks: Optional[Greeks] = None
    
    # Account fit
    margin_required: float = 0.0
    fits_account: bool = False
    
    # Scores
    pop_score: float = 0.0
    ev_score: float = 0.0
    income_score: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.strategy_type.upper()} on {self.ticker} @ ${self.underlying_price:.2f}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'ticker': self.ticker,
            'strategy_type': self.strategy_type,
            'underlying_price': self.underlying_price,
            'legs': [
                {
                    'strike': l.strike,
                    'expiration': l.expiration,
                    'dte': l.dte,
                    'premium': l.premium,
                    'option_type': l.option_type,
                    'action': l.action
                }
                for l in self.legs
            ],
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'breakevens': self.breakevens,
            'pop': self.pop,
            'expected_value': self.expected_value,
            'risk_adjusted_return': self.risk_adjusted_return,
            'margin_required': self.margin_required,
            'fits_account': self.fits_account
        }


def validate_strategy_risk(strategy: MultiLegStrategy) -> Tuple[bool, str]:
    """
    Validate that a strategy has defined, finite risk.
    
    Returns (is_valid, reason).
    Checks for:
    - Infinite risk (naked shorts, ratio spreads)
    - Undefined P&L (zero/negative max_profit or max_loss)
    - Invalid spread construction
    """
    legs = strategy.legs
    if not legs:
        return False, "no legs"

    # --- Check for naked positions (short without matching long) ---
    short_calls = [l for l in legs if l.action == 'sell' and l.option_type == 'call']
    long_calls = [l for l in legs if l.action == 'buy' and l.option_type == 'call']
    short_puts = [l for l in legs if l.action == 'sell' and l.option_type == 'put']
    long_puts = [l for l in legs if l.action == 'buy' and l.option_type == 'put']

    short_call_qty = sum(l.quantity for l in short_calls)
    long_call_qty = sum(l.quantity for l in long_calls)
    short_put_qty = sum(l.quantity for l in short_puts)
    long_put_qty = sum(l.quantity for l in long_puts)

    # Naked short calls = infinite risk
    if short_call_qty > 0 and long_call_qty == 0:
        return False, "naked short call(s) — infinite risk"
    # Naked short puts = undefined risk (strike * 100)
    if short_put_qty > 0 and long_put_qty == 0:
        return False, "naked short put(s) — undefined risk"

    # --- Ratio spreads (more shorts than longs) ---
    if short_call_qty > long_call_qty:
        return False, f"ratio call spread ({short_call_qty}:{long_call_qty} short:long) — undefined risk"
    if short_put_qty > long_put_qty:
        return False, f"ratio put spread ({short_put_qty}:{long_put_qty} short:long) — undefined risk"

    # --- P&L validation ---
    if strategy.max_profit < 0:
        return False, f"negative max_profit ({strategy.max_profit:.2f})"
    if strategy.max_loss < 0:
        return False, f"negative max_loss ({strategy.max_loss:.2f})"
    if strategy.max_profit == 0:
        return False, "zero max_profit — breakeven-only trade"
    if strategy.max_loss == 0:
        return False, "zero max_loss — likely bad data or arbitrage"

    # --- Breakeven validation ---
    if not strategy.breakevens or any(b <= 0 or not np.isfinite(b) for b in strategy.breakevens):
        return False, f"invalid breakeven(s): {strategy.breakevens}"

    # --- Finite check on all numeric fields ---
    for field_name, val in [('max_profit', strategy.max_profit),
                             ('max_loss', strategy.max_loss),
                             ('pop', strategy.pop),
                             ('expected_value', strategy.expected_value),
                             ('risk_adjusted_return', strategy.risk_adjusted_return)]:
        if not np.isfinite(val):
            return False, f"non-finite {field_name}: {val}"

    return True, "ok"


class LegOptimizer:
    """
    Optimize multi-leg option strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.045, account_total: float = DEFAULT_ACCOUNT_TOTAL):
        self.bs = BlackScholes()
        self.calc = ProbabilityCalculator(risk_free_rate)
        self.analyzer = ChainAnalyzer()
        self.account_total = account_total
    
    # Minimum IV floor — pre-market/post-market data often reports near-zero IV
    # which produces 100% POP and meaningless greeks.  15% is a conservative
    # floor (VIX rarely stays below 12 for extended periods).
    IV_FLOOR = 0.15

    def calculate_strategy_metrics(self, strategy: MultiLegStrategy,
                                   iv: float = 0.25) -> MultiLegStrategy:
        """
        Calculate all metrics for a strategy
        """
        if not strategy.legs:
            return strategy

        # Enforce IV floor for realistic probability/greeks calculations
        # (pre-market/post-market data often reports near-zero IV)
        iv = max(iv, self.IV_FLOOR)
        
        # Calculate net premium (per share)
        net_premium = sum(leg.net_premium for leg in strategy.legs)
        
        # Get strategy type and calculate max P/L
        if strategy.strategy_type == 'put_credit_spread':
            # Sell higher strike put, buy lower strike put
            short_leg = [l for l in strategy.legs if l.action == 'sell' and l.option_type == 'put'][0]
            long_leg = [l for l in strategy.legs if l.action == 'buy' and l.option_type == 'put'][0]
            
            width = short_leg.strike - long_leg.strike
            strategy.max_profit = net_premium * 100  # Per contract
            strategy.max_loss = max(0, (width - net_premium) * 100)  # Width minus credit, min $0
            strategy.breakevens = [short_leg.strike - net_premium]
            
        elif strategy.strategy_type == 'call_credit_spread':
            # Sell lower strike call, buy higher strike call
            short_leg = [l for l in strategy.legs if l.action == 'sell' and l.option_type == 'call'][0]
            long_leg = [l for l in strategy.legs if l.action == 'buy' and l.option_type == 'call'][0]
            
            width = long_leg.strike - short_leg.strike
            strategy.max_profit = net_premium * 100
            strategy.max_loss = max(0, (width - net_premium) * 100)  # Width minus credit, min $0
            strategy.breakevens = [short_leg.strike + net_premium]
            
        elif strategy.strategy_type == 'iron_condor':
            # Short put spread + short call spread
            puts = [l for l in strategy.legs if l.option_type == 'put']
            calls = [l for l in strategy.legs if l.option_type == 'call']
            
            put_short = [l for l in puts if l.action == 'sell'][0]
            put_long = [l for l in puts if l.action == 'buy'][0]
            call_short = [l for l in calls if l.action == 'sell'][0]
            call_long = [l for l in calls if l.action == 'buy'][0]
            
            put_width = put_short.strike - put_long.strike
            call_width = call_long.strike - call_short.strike
            max_width = max(put_width, call_width)
            
            strategy.max_profit = net_premium * 100
            strategy.max_loss = (max_width - net_premium) * 100
            strategy.breakevens = [put_short.strike - net_premium, call_short.strike + net_premium]
            
        elif strategy.strategy_type in ['put_debit_spread', 'call_debit_spread']:
            # Debit spreads
            if strategy.strategy_type == 'put_debit_spread':
                long_leg = [l for l in strategy.legs if l.action == 'buy' and l.option_type == 'put'][0]
                short_leg = [l for l in strategy.legs if l.action == 'sell' and l.option_type == 'put'][0]
                width = long_leg.strike - short_leg.strike
            else:
                long_leg = [l for l in strategy.legs if l.action == 'buy' and l.option_type == 'call'][0]
                short_leg = [l for l in strategy.legs if l.action == 'sell' and l.option_type == 'call'][0]
                width = short_leg.strike - long_leg.strike
            
            strategy.max_profit = (width + net_premium) * 100  # net_premium is negative for debit
            strategy.max_loss = -net_premium * 100
            
        else:
            # Default: sum of premiums
            strategy.max_profit = abs(net_premium) * 100
            strategy.max_loss = abs(net_premium) * 100
        
        # --- Guard: skip strategies with non-positive P&L ---
        if strategy.max_profit <= 0 or strategy.max_loss <= 0:
            logger.debug("Skipping strategy: max_profit=%.2f, max_loss=%.2f",
                         strategy.max_profit, strategy.max_loss)
            return strategy
        
        # Calculate POP
        T = max(strategy.legs[0].dte, 1) / 365.0  # Ensure minimum 1 day
        S = strategy.underlying_price
        
        if strategy.strategy_type == 'put_credit_spread':
            strategy.pop = self.calc.pop_vertical_spread(
                S, short_leg.strike, long_leg.strike, T, iv,
                net_premium, 'put_credit'
            )
        elif strategy.strategy_type == 'call_credit_spread':
            strategy.pop = self.calc.pop_vertical_spread(
                S, short_leg.strike, long_leg.strike, T, iv,
                net_premium, 'call_credit'
            )
        elif strategy.strategy_type == 'iron_condor':
            # Get breakevens for Monte Carlo simulation
            lower_breakeven = strategy.breakevens[0] if strategy.breakevens else (put_short.strike - net_premium)
            upper_breakeven = strategy.breakevens[1] if len(strategy.breakevens) > 1 else (call_short.strike + net_premium)

            # Use Monte Carlo simulation for more accurate POP calculation
            # Monte Carlo simulates price paths and checks if price stays within
            # bounds at ANY point ("touch" probability), which matches how
            # brokers like Tastytrade calculate POP for Iron Condors.
            # Black-Scholes only calculates probability AT expiration.
            try:
                mc_pop = self.calc.monte_carlo_pop_iron_condor(
                    S, lower_breakeven, upper_breakeven, T, iv,
                    n_sims=100000, steps_per_day=2
                )
                strategy.pop = mc_pop
            except Exception as e:
                logger.debug(f"Monte Carlo POP failed, falling back to Black-Scholes: {e}")
                # Fallback to Black-Scholes closed-form solution
                # Correct POP for Iron Condor: P(Lower BE < S_T < Upper BE)
                # This is P(S_T < Upper BE) - P(S_T < Lower BE)
                # pop_vertical_spread(call_credit) returns P(S_T < Upper BE)
                # pop_vertical_spread(put_credit) returns P(S_T > Lower BE)
                put_pop = self.calc.pop_vertical_spread(
                    S, put_short.strike, put_long.strike, T, iv,
                    net_premium, 'put_credit'
                )
                call_pop = self.calc.pop_vertical_spread(
                    S, call_short.strike, call_long.strike, T, iv,
                    net_premium, 'call_credit'
                )
                # P(S_T < Lower BE) = 1 - put_pop
                # So POP = call_pop - (1 - put_pop) = call_pop + put_pop - 1
                strategy.pop = max(0, put_pop + call_pop - 1)
        elif strategy.strategy_type == 'put_debit_spread':
             strategy.pop = self.calc.pop_vertical_spread(S, long_leg.strike, short_leg.strike, T, iv, net_premium, 'put_debit')
        elif strategy.strategy_type == 'call_debit_spread':
             strategy.pop = self.calc.pop_vertical_spread(S, long_leg.strike, short_leg.strike, T, iv, net_premium, 'call_debit')
        else:
            strategy.pop = 0.5
            
        # Expected Value (guard: POP must be in [0,1])
        pop_clamped = max(0.0, min(1.0, strategy.pop))
        strategy.pop = pop_clamped
        strategy.expected_value = self.calc.expected_value(
            pop_clamped, strategy.max_profit, strategy.max_loss
        )
        
        # Risk-adjusted return (annualized EV / risk)
        if strategy.max_loss > 0:
            annual_factor = 365.0 / max(strategy.legs[0].dte, 1)
            raw_return = strategy.expected_value / strategy.max_loss * annual_factor
            # Cap at reasonable bounds to avoid absurd numbers
            strategy.risk_adjusted_return = max(-10.0, min(10.0, raw_return))
        else:
            strategy.risk_adjusted_return = 0.0
        
        # Margin requirement (simplified)
        strategy.margin_required = strategy.max_loss
        
        # Check account fit
        strategy.fits_account = fits_account_constraints(
            strategy.max_loss, strategy.margin_required, self.account_total
        )
        
        # Calculate total Greeks
        total_delta = sum(
            (leg.greeks.delta if leg.greeks else 0) * leg.quantity * (1 if leg.action == 'buy' else -1)
            for leg in strategy.legs
        ) if any(leg.greeks for leg in strategy.legs) else 0
        
        total_theta = sum(
            (leg.greeks.theta if leg.greeks else 0) * leg.quantity * (1 if leg.action == 'buy' else -1)
            for leg in strategy.legs
        ) if any(leg.greeks for leg in strategy.legs) else 0
        
        strategy.total_greeks = Greeks(
            delta=total_delta,
            gamma=0,  # Simplified
            theta=total_theta,
            vega=0,
            rho=0
        )
        
        return strategy
    
    def score_strategies(self, strategies: List[MultiLegStrategy], 
                         mode: str = 'ev') -> List[MultiLegStrategy]:
        """
        Score and sort strategies based on specified mode
        
        Args:
            strategies: List of MultiLegStrategy objects
            mode: Scoring mode - 'pop' (maximize POP), 'ev' (maximize expected value),
                  'income' (maximize theta/income)
        
        Returns:
            Sorted list of strategies (highest score first)
        """
        # Ensure all strategies have metrics calculated
        for strategy in strategies:
            if strategy.pop == 0 and strategy.legs:
                # Need to calculate metrics - use default IV
                self.calculate_strategy_metrics(strategy)
        
        # Sort based on mode
        if mode == 'pop':
            # Sort by POP (highest first), then by max_loss (lowest first)
            scored = sorted(strategies, 
                          key=lambda s: (s.pop, -s.max_loss), 
                          reverse=True)
        elif mode == 'ev':
            # Sort by EV score (already calculated in metrics)
            scored = sorted(strategies, 
                          key=lambda s: (s.ev_score, s.pop), 
                          reverse=True)
        elif mode == 'income':
            # Sort by income score (theta-based)
            scored = sorted(strategies, 
                          key=lambda s: (s.income_score, s.pop), 
                          reverse=True)
        else:
            # Default to EV sorting
            scored = sorted(strategies, 
                          key=lambda s: s.ev_score, 
                          reverse=True)
        
        return scored
    
    def optimize_vertical_spreads(self, chain: OptionChain, 
                                  spread_type: str = 'put_credit',
                                  max_width: float = 5.0,
                                  min_dte: int = 7,
                                  max_dte: int = 45) -> List[MultiLegStrategy]:
        """
        Find optimal vertical spreads from options chain
        
        spread_type: 'put_credit', 'call_credit', 'put_debit', 'call_debit'
        """
        strategies = []
        
        if spread_type in ['put_credit', 'put_debit']:
            options = chain.puts
            opt_type = 'put'
        else:
            options = chain.calls
            opt_type = 'call'
        
        if len(options) < 2:
            return strategies
        
        S = chain.underlying_price
        T = chain.dte / 365.0
        r = 0.045
        
        # Get widths to try
        # $1-wide spreads are critical for high-priced underlyings (SPY, QQQ)
        # where OTM credit spreads need narrow widths to fit small accounts
        # Wider spreads (7, 10, 15, 20) useful for larger accounts seeking better credit
        widths = [w for w in [1, 2, 3, 5, 7, 10, 15, 20] if w <= max_width]
        
        for width in widths:
            # Try different short strikes
            for i, short_opt in enumerate(options):
                # OTM/ATM validation: reject deep ITM short strikes
                # For credit spreads, short strike must be OTM or near ATM
                if spread_type == 'put_credit':
                    # Short put must be at or below current price (OTM/ATM)
                    # Allow small buffer (2% ITM) for near-ATM strikes
                    if short_opt['strike'] > S * 1.02:
                        continue
                elif spread_type == 'call_credit':
                    # Short call must be at or above current price (OTM/ATM)
                    if short_opt['strike'] < S * 0.98:
                        continue
                
                # Liquidity filter: require valid bid/ask OR lastPrice fallback
                # Pre-market / post-market: bid/ask are often 0 but lastPrice exists
                has_bid_ask = short_opt.get('has_valid_bid_ask', False)
                if not has_bid_ask and short_opt['mid_price'] <= 0:
                    logger.debug("Skipping short strike %.0f: no bid/ask and no lastPrice", short_opt['strike'])
                    continue
                if short_opt['mid_price'] <= 0.05:
                    continue
                # Only apply spread width filter when we have real bid/ask data
                if has_bid_ask and short_opt['spread_pct'] > 0.20:
                    logger.debug("Skipping short strike %.0f: wide spread %.1f%%",
                                 short_opt['strike'], short_opt['spread_pct'] * 100)
                    continue
                
                # Require non-zero bid for a real market (avoid $0.00 bid / $0.01 ask phantom quotes)
                bid = short_opt.get('bid', 0) or 0
                if bid <= 0:
                    logger.debug("Skipping short strike %.0f: zero bid (no real market)", short_opt['strike'])
                    continue
                
                # Volume/Open Interest filter - avoid phantom quotes with no actual market
                volume = short_opt.get('volume', 0) or 0
                oi = short_opt.get('open_interest', 0) or 0
                if volume == 0 and oi == 0:
                    logger.debug("Skipping short strike %.0f: zero volume and open interest", short_opt['strike'])
                    continue
                
                # For $1-wide spreads on expensive underlyings, skip deep OTM 
                # where net credit would be negligible (< $0.05/share)
                # This is checked after pairing below
                
                # Find matching long strike
                if spread_type in ['put_credit', 'put_debit']:
                    target_long_strike = short_opt['strike'] - width
                else:
                    target_long_strike = short_opt['strike'] + width
                
                # Find closest long option
                long_opt = None
                long_idx = None
                min_diff = float('inf')
                
                for j, opt in enumerate(options):
                    diff = abs(opt['strike'] - target_long_strike)
                    if diff < min_diff:
                        min_diff = diff
                        long_opt = opt
                        long_idx = j
                
                if not long_opt or min_diff > 0.5:
                    continue
                
                # Skip if same strike
                if short_opt['strike'] == long_opt['strike']:
                    continue
                
                # Volume/Open Interest filter for long strike
                long_volume = long_opt.get('volume', 0) or 0
                long_oi = long_opt.get('open_interest', 0) or 0
                if long_volume == 0 and long_oi == 0:
                    logger.debug("Skipping long strike %.0f: zero volume and open interest", long_opt['strike'])
                    continue
                
                # Require non-zero bid for long strike too
                long_bid = long_opt.get('bid', 0) or 0
                if long_bid <= 0:
                    logger.debug("Skipping long strike %.0f: zero bid (no real market)", long_opt['strike'])
                    continue
                
                # Calculate implied vol for Black-Scholes
                # Use ATM IV as estimate
                atm_idx = min(range(len(options)), 
                             key=lambda k: abs(options[k]['strike'] - S))
                iv = max(options[atm_idx]['implied_vol'] or 0.20, 0.15)
                
                # Calculate Greeks for both legs
                short_greeks = self.bs.calculate_greeks(
                    S, short_opt['strike'], T, r, 
                    max(short_opt['implied_vol'] or iv, 0.15), opt_type
                )
                long_greeks = self.bs.calculate_greeks(
                    S, long_opt['strike'], T, r,
                    max(long_opt['implied_vol'] or iv, 0.15), opt_type
                )
                
                # Use mid-point pricing for realistic fill estimates:
                # Most limit orders fill near the mid-point, not at bid/ask extremes.
                # Conservative bid/ask pricing understates credit spreads and
                # overstates debit spreads, skewing EV and POP comparisons.
                short_premium = short_opt['mid_price']
                long_premium = long_opt['mid_price']

                # Skip if mid-point is 0 (no real market)
                if short_premium <= 0 or long_premium <= 0:
                    logger.debug("Skipping %s/%s: short_mid=%.2f, long_mid=%.2f (no market)",
                                 short_opt['strike'], long_opt['strike'], short_premium, long_premium)
                    continue

                # Build legs based on spread type
                if spread_type == 'put_credit':
                    legs = [
                        TradeLeg(
                            strike=short_opt['strike'],
                            expiration=chain.expiration_date,
                            dte=chain.dte,
                            premium=short_premium,
                            option_type='put',
                            action='sell',
                            greeks=short_greeks
                        ),
                        TradeLeg(
                            strike=long_opt['strike'],
                            expiration=chain.expiration_date,
                            dte=chain.dte,
                            premium=long_premium,
                            option_type='put',
                            action='buy',
                            greeks=long_greeks
                        )
                    ]
                    strategy_type = 'put_credit_spread'
                elif spread_type == 'call_credit':
                    legs = [
                        TradeLeg(
                            strike=short_opt['strike'],
                            expiration=chain.expiration_date,
                            dte=chain.dte,
                            premium=short_premium,
                            option_type='call',
                            action='sell',
                            greeks=short_greeks
                        ),
                        TradeLeg(
                            strike=long_opt['strike'],
                            expiration=chain.expiration_date,
                            dte=chain.dte,
                            premium=long_premium,
                            option_type='call',
                            action='buy',
                            greeks=long_greeks
                        )
                    ]
                    strategy_type = 'call_credit_spread'
                else:
                    # Debit spreads - reverse actions
                    continue  # Skip for now, focus on credit spreads
                
                # Minimum net credit filter: skip if credit < $0.05/share
                # (avoids deep OTM $1 spreads with $1-2 total credit for $98-99 risk)
                net_credit = sum(l.net_premium for l in legs)
                if net_credit < 0.05:
                    continue
                
                # Minimum credit-to-width ratio: at least 10% of width
                # Ensures reasonable risk/reward (e.g., $0.20 credit on $2 width)
                actual_width = abs(legs[0].strike - legs[1].strike)
                if actual_width > 0 and net_credit / actual_width < 0.10:
                    continue
                
                strategy = MultiLegStrategy(
                    ticker=chain.ticker,
                    strategy_type=strategy_type,
                    underlying_price=S,
                    legs=legs
                )
                
                strategy = self.calculate_strategy_metrics(strategy, iv)
                
                # Skip unrealistic scenarios (credit >= width means no risk, likely bad data)
                if strategy.max_loss <= 0:
                    continue
                
                # Only include if it fits account (use 50% of account as max risk ceiling)
                # This allows larger spreads for larger accounts while keeping risk managed
                account_max_risk = self.account_total * 0.50
                if strategy.max_loss <= account_max_risk:
                    strategies.append(strategy)
        
        return strategies
    
    def optimize_iron_condors(self, chain: OptionChain,
                             put_width: float = 5.0,
                             call_width: float = 5.0,
                             otm_target: float = 0.10) -> List[MultiLegStrategy]:
        """
        Find optimal iron condors
        
        otm_target: Target delta for short options (default 10 delta ~ 10% OTM)
        """
        strategies = []
        
        if not chain.puts or not chain.calls:
            return strategies
        
        S = chain.underlying_price
        T = chain.dte / 365.0
        r = 0.045
        
        # Find 10% OTM strikes
        put_target = S * (1 - otm_target)
        call_target = S * (1 + otm_target)
        
        # Find closest puts
        put_short = None
        put_long = None
        call_short = None
        call_long = None
        
        for put in chain.puts:
            if put['strike'] <= put_target and not put_short:
                put_short = put
        for put in chain.puts:
            if put_short and put['strike'] == put_short['strike'] - put_width:
                put_long = put
                break
        
        for call in chain.calls:
            if call['strike'] >= call_target and not call_short:
                call_short = call
        for call in chain.calls:
            if call_short and call['strike'] == call_short['strike'] + call_width:
                call_long = call
                break
        
        if not all([put_short, put_long, call_short, call_long]):
            return strategies
        
        # Get IV from the short strikes (higher due to skew/vol smile)
        put_short_iv = next((p['implied_vol'] for p in chain.puts if abs(p['strike'] - put_short['strike']) < 0.01), None)
        call_short_iv = next((c['implied_vol'] for c in chain.calls if abs(c['strike'] - call_short['strike']) < 0.01), None)
        
        # Use the maximum of the two short strike IVs to be conservative
        # (OTM puts typically have much higher IV than ATM due to skew)
        iv = max(put_short_iv or 0.20, call_short_iv or 0.20, self.IV_FLOOR)
        
        # Use mid-point pricing for realistic fill estimates
        ps_prem = put_short['mid_price']
        pl_prem = put_long['mid_price']
        cs_prem = call_short['mid_price']
        cl_prem = call_long['mid_price']

        # Calculate Greeks
        legs = [
            TradeLeg(put_short['strike'], chain.expiration_date, chain.dte,
                    ps_prem, 'put', 'sell',
                    greeks=self.bs.calculate_greeks(S, put_short['strike'], T, r, iv, 'put')),
            TradeLeg(put_long['strike'], chain.expiration_date, chain.dte,
                    pl_prem, 'put', 'buy',
                    greeks=self.bs.calculate_greeks(S, put_long['strike'], T, r, iv, 'put')),
            TradeLeg(call_short['strike'], chain.expiration_date, chain.dte,
                    cs_prem, 'call', 'sell',
                    greeks=self.bs.calculate_greeks(S, call_short['strike'], T, r, iv, 'call')),
            TradeLeg(call_long['strike'], chain.expiration_date, chain.dte,
                    cl_prem, 'call', 'buy',
                    greeks=self.bs.calculate_greeks(S, call_long['strike'], T, r, iv, 'call'))
        ]
        
        strategy = MultiLegStrategy(
            ticker=chain.ticker,
            strategy_type='iron_condor',
            underlying_price=S,
            legs=legs
        )
        
        strategy = self.calculate_strategy_metrics(strategy, iv)
        
        if strategy.max_loss <= MAX_RISK_PER_TRADE * 1.5:
            strategies.append(strategy)
        
        return strategies
