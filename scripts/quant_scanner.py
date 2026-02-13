#!/usr/bin/env python3
"""
Quantitative Options Scanner

A mathematically-rigorous options scanner built from first principles.
Replaces the technical-indicator-heavy conviction engine with options-first analysis.

Features:
- Options chain analysis (IV surfaces, skew, term structure)
- Multi-leg strategy optimization (spreads, iron condors, butterflies, calendars)
- Black-Scholes/Monte Carlo POP calculations
- Expected Value and risk-adjusted return scoring
- Account-aware filtering for small accounts

Usage:
    quant_scanner.py SPY --mode pop
    quant_scanner.py AAPL TSLA NVDA --mode income --max-loss 100
    quant_scanner.py SPY --mode ev --dte 30

Account Constraints:
- Total Account: $500 (default, override with --account)
- Max Risk Per Trade: $100
- Min Cash Buffer: $150
- Available Capital: Account - Buffer
"""

import argparse
import sys
import json
from typing import List, Optional
from datetime import datetime

from options_math import (
    BlackScholes, ProbabilityCalculator, VolatilityAnalyzer,
    fits_account_constraints, MAX_RISK_PER_TRADE, DEFAULT_ACCOUNT_TOTAL,
    ACCOUNT_TOTAL, AVAILABLE_CAPITAL, MIN_CASH_BUFFER
)
from chain_analyzer import ChainFetcher, ChainAnalyzer, OptionChain
from leg_optimizer import LegOptimizer, MultiLegStrategy, validate_strategy_risk


class QuantScanner:
    """
    Main quantitative options scanner
    """
    
    def __init__(self, account_total: float = DEFAULT_ACCOUNT_TOTAL,
                 max_risk_per_trade: float = MAX_RISK_PER_TRADE,
                 min_cash_buffer: float = MIN_CASH_BUFFER):
        self.account_total = account_total
        self.max_risk_per_trade = max_risk_per_trade
        self.min_cash_buffer = min_cash_buffer
        self.fetcher = ChainFetcher(rate_limit_delay=0.3)
        self.analyzer = ChainAnalyzer(self.fetcher)
        self.optimizer = LegOptimizer(account_total=account_total,
                                      max_risk_per_trade=max_risk_per_trade,
                                      min_cash_buffer=min_cash_buffer)
        self.vol_analyzer = VolatilityAnalyzer()
    
    def scan_ticker(self, ticker: str, mode: str = 'pop',
                   min_dte: int = 7, max_dte: int = 45,
                   max_loss_limit: float = MAX_RISK_PER_TRADE,
                   min_pop: float = 0.0,
                   min_width: float = 1.0,
                   max_width: float = 5.0,
                   verbose: bool = False) -> Optional[dict]:
        """
        Scan a single ticker and return best strategies
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scanning {ticker} | Mode: {mode.upper()}")
            print(f"{'='*60}")
        
        # Fetch quote
        quote = self.fetcher.fetch_quote(ticker)
        if not quote:
            if verbose:
                print(f"ERROR: Could not fetch quote for {ticker}")
            return None
        
        price = quote.get('regularMarketPrice', 0)
        if price == 0:
            if verbose:
                print(f"ERROR: No price data for {ticker}")
            return None
        
        market_state = quote.get('marketState', 'UNKNOWN')
        
        if verbose:
            print(f"\nCurrent Price: ${price:.2f}")
            change = quote.get('regularMarketChange', 0)
            change_pct = quote.get('regularMarketChangePercent', 0)
            print(f"Change: ${change:+.2f} ({change_pct:+.2f}%)")
            if market_state not in ('REGULAR',):
                print(f"  ⚠ Market: {market_state} — bid/ask may be stale/wider than during market hours")
        
        # Fetch multiple expiration chains
        chains = self.fetcher.fetch_multiple_expirations(
            ticker, num_expirations=4, min_dte=min_dte, max_dte=max_dte
        )
        
        if not chains:
            if verbose:
                print(f"ERROR: No options chains available for {ticker}")
            return None
        
        if verbose:
            print(f"\nFound {len(chains)} expiration dates:")
            for chain in chains:
                print(f"  - {chain.expiration_date[:10]} ({chain.dte} DTE)")
        
        # Analyze chains and find strategies
        all_strategies = []
        
        for chain in chains:
            # Skip if too illiquid
            liquidity = self.analyzer.analyze_liquidity(chain)
            if liquidity['score'] < 30:
                if verbose:
                    print(f"  Skipping {chain.dte} DTE - poor liquidity")
                continue
            
            # Find vertical spreads
            put_spreads = self.optimizer.optimize_vertical_spreads(
                chain, spread_type='put_credit', max_width=max_width
            )
            call_spreads = self.optimizer.optimize_vertical_spreads(
                chain, spread_type='call_credit', max_width=max_width
            )
            
            # Find iron condors
            condors = self.optimizer.optimize_iron_condors(chain)
            
            all_strategies.extend(put_spreads)
            all_strategies.extend(call_spreads)
            all_strategies.extend(condors)
        
        if not all_strategies:
            if verbose:
                print(f"\nNo viable strategies found for {ticker}")
            return None
        
        # Validate: reject infinite/undefined risk strategies
        validated = []
        for s in all_strategies:
            is_valid, reason = validate_strategy_risk(s)
            if is_valid:
                validated.append(s)
            elif verbose:
                print(f"  REJECTED: {s.strategy_type} — {reason}")
        all_strategies = validated
        
        if not all_strategies:
            if verbose:
                print(f"\nNo valid strategies after risk validation for {ticker}")
            return None
        
        # Filter by minimum spread width
        if min_width > 1.0:
            width_filtered = []
            for s in all_strategies:
                strikes = [leg.strike for leg in s.legs]
                if len(strikes) >= 2:
                    width = max(strikes) - min(strikes)
                    if width >= min_width:
                        width_filtered.append(s)
                else:
                    width_filtered.append(s)
            if verbose:
                print(f"  Width filter (>= ${min_width:.0f}): {len(all_strategies)} → {len(width_filtered)}")
            all_strategies = width_filtered
        
        if not all_strategies:
            if verbose:
                print(f"\nNo strategies after width filter for {ticker}")
            return None
        
        # Filter by maximum spread width
        if max_width < 5.0:
            max_width_filtered = []
            for s in all_strategies:
                strikes = [leg.strike for leg in s.legs]
                if len(strikes) >= 2:
                    width = max(strikes) - min(strikes)
                    if width <= max_width:
                        max_width_filtered.append(s)
                else:
                    max_width_filtered.append(s)
            if verbose:
                print(f"  Max width filter (<= ${max_width:.0f}): {len(all_strategies)} → {len(max_width_filtered)}")
            all_strategies = max_width_filtered
        
        if not all_strategies:
            if verbose:
                print(f"\nNo strategies after max width filter for {ticker}")
            return None
        
        if verbose:
            print(f"\nFound {len(all_strategies)} validated strategies")
        
        # Score strategies based on mode
        scored = self.optimizer.score_strategies(all_strategies, mode=mode)
        
        # Filter by max loss and min POP
        fitting_strategies = [
            s for s in scored 
            if s.max_loss <= max_loss_limit and s.pop >= min_pop
        ]
        
        if not fitting_strategies:
            if min_pop > 0:
                # User explicitly requested min POP — don't fall back to unfiltered
                if verbose:
                    print(f"\nNo strategies meet min POP {min_pop*100:.0f}% and max loss ${max_loss_limit:.0f}")
                return None
            # Only fall back when no min_pop filter was set
            fitting_strategies = sorted(scored, key=lambda x: x.max_loss)[:3]
        
        # Take top 3
        top_strategies = fitting_strategies[:3]
        
        if verbose:
            self._print_strategies(ticker, price, top_strategies, mode)
        
        return {
            'ticker': ticker,
            'price': price,
            'mode': mode,
            'strategies': [s.to_dict() for s in top_strategies],
            'total_found': len(all_strategies),
            'fitting_count': len(fitting_strategies)
        }
    
    def _print_strategies(self, ticker: str, price: float, 
                         strategies: List[MultiLegStrategy], mode: str):
        """Pretty print strategy results"""
        
        print(f"\n{'─'*60}")
        print(f"TOP STRATEGIES FOR {ticker} (Mode: {mode.upper()})")
        print(f"{'─'*60}")
        
        for i, s in enumerate(strategies, 1):
            print(f"\n{'▓'*60}")
            print(f"STRATEGY #{i}: {s.strategy_type.upper()}")
            print(f"{'▓'*60}")
            
            print(f"  Expiration: {s.legs[0].expiration[:10]} ({s.legs[0].dte} DTE)")
            
            print(f"\n  LEGS:")
            for leg in s.legs:
                action = "SELL" if leg.action == 'sell' else "BUY"
                print(f"    {action:4} {leg.option_type.upper():4} @ ${leg.strike:7.2f} "
                      f"| Premium: ${leg.premium:.2f} (mid) | DTE: {leg.dte}")
            
            print(f"\n  P&L PROFILE:")
            print(f"    Max Profit:     ${s.max_profit:7.2f}")
            print(f"    Max Loss:       ${s.max_loss:7.2f} {'✓ FITS' if s.fits_account else '✗ TOO RISKY'}")
            print(f"    Breakeven(s):   {', '.join(f'${b:.2f}' for b in s.breakevens)}")
            
            print(f"\n  PROBABILITY & VALUE:")
            print(f"    Probability of Profit (POP): {s.pop*100:5.1f}%")
            print(f"    Expected Value (EV):         ${s.expected_value:+.2f}")
            print(f"    Risk-Adjusted Return:        {s.risk_adjusted_return:+.2f}")
            
            if s.total_greeks:
                print(f"\n  GREEKS (Per Contract):")
                print(f"    Delta: {s.total_greeks.delta:+.3f}")
                print(f"    Theta: ${s.total_greeks.theta:+.2f}/day")
            
            # Recommendation
            if s.fits_account and s.pop > 0.6 and s.expected_value > 0:
                print(f"\n  ★ RECOMMENDATION: EXECUTE")
            elif s.fits_account and s.pop > 0.5:
                print(f"\n  ○ RECOMMENDATION: CONSIDER")
            else:
                print(f"\n  ✗ RECOMMENDATION: PASS")
        
        print(f"\n{'='*60}")
    
    def scan_multiple(self, tickers: List[str], mode: str = 'pop',
                     min_dte: int = 7, max_dte: int = 45,
                     max_loss_limit: float = MAX_RISK_PER_TRADE,
                     min_pop: float = 0.0,
                     min_width: float = 1.0,
                     max_width: float = 5.0,
                     json_output: bool = False) -> List[dict]:
        """
        Scan multiple tickers and return aggregated results
        """
        results = []
        
        if not json_output:
            print(f"\n{'#'*70}")
            print(f"# QUANTITATIVE OPTIONS SCANNER")
            print(f"# Mode: {mode.upper()}")
            print(f"# Tickers: {', '.join(tickers)}")
            print(f"# Account: ${self.account_total} | Max Risk: ${max_loss_limit}")
            print(f"# DTE Range: {min_dte}-{max_dte}")
            if min_pop > 0:
                print(f"# Min POP: {min_pop*100:.0f}%")
            if min_width > 1:
                print(f"# Min Width: ${min_width:.0f}")
            if max_width < 5:
                print(f"# Max Width: ${max_width:.0f}")
            print(f"{'#'*70}\n")
        
        for ticker in tickers:
            result = self.scan_ticker(
                ticker, mode, min_dte, max_dte, max_loss_limit, min_pop,
                min_width=min_width, max_width=max_width, verbose=not json_output
            )
            if result:
                results.append(result)
        
        # Summary
        if not json_output:
            print(f"\n{'#'*70}")
            print(f"# SCAN SUMMARY")
            print(f"{'#'*70}")
            print(f"\nTickers Scanned: {len(tickers)}")
            print(f"Tickers with Opportunities: {len(results)}")
            print(f"\nAccount Constraints:")
            available = self.account_total - self.min_cash_buffer
            print(f"  Total Account: ${self.account_total}")
            print(f"  Max Risk/Trade: ${max_loss_limit}")
            print(f"  Min Cash Buffer: ${self.min_cash_buffer}")
            print(f"  Available Capital: ${available}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Quantitative Options Scanner - Mathematical options analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s SPY --mode pop                    # Maximize POP for SPY
  %(prog)s AAPL TSLA --mode ev               # Max expected value
  %(prog)s SPY QQQ --mode income             # Income/theta plays
  %(prog)s NVDA --mode earnings              # Earnings/vol crush plays
  %(prog)s SPY --json                        # Machine-readable output
  %(prog)s SPY --min-pop 75                  # Only trades with 75%%+ POP
  %(prog)s SPY --min-width 2 --min-pop 75     # $2+ wide spreads, 75%%+ POP
  %(prog)s SPY --min-width 2 --max-width 3   # Only $2-3 wide spreads
  %(prog)s SPY --dte 14 --max-loss 50        # Custom DTE and risk
        """
    )
    
    parser.add_argument('tickers', nargs='+', 
                       help='Stock ticker(s) to scan')
    
    parser.add_argument('--mode', '-m', 
                       choices=['pop', 'ev', 'income', 'earnings'],
                       default='pop',
                       help='Scanning mode (default: pop)')
    
    parser.add_argument('--min-dte', type=int, default=7,
                       help='Minimum days to expiration (default: 7)')
    
    parser.add_argument('--max-dte', type=int, default=45,
                       help='Maximum days to expiration (default: 45)')
    
    parser.add_argument('--max-loss', type=float, default=MAX_RISK_PER_TRADE,
                       help=f'Maximum loss per trade (default: ${MAX_RISK_PER_TRADE})')
    
    parser.add_argument('--min-pop', type=float, default=0.0,
                       help='Minimum Probability of Profit %% (default: 0)')
    
    parser.add_argument('--min-width', type=int, default=1,
                       help='Minimum spread width in dollars (default: 1)')
    
    parser.add_argument('--max-width', type=int, default=5,
                       help='Maximum spread width in dollars (default: 5)')
    
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output JSON format')
    
    parser.add_argument('--account', type=float, default=DEFAULT_ACCOUNT_TOTAL,
                       help=f'Total account balance (default: ${DEFAULT_ACCOUNT_TOTAL:.0f})')
    
    parser.add_argument('--max-risk', type=float, default=MAX_RISK_PER_TRADE,
                       help=f'Maximum risk per trade in dollars (default: ${MAX_RISK_PER_TRADE:.0f})')
    
    parser.add_argument('--min-cash', type=float, default=MIN_CASH_BUFFER,
                       help=f'Minimum cash buffer to keep in reserve (default: ${MIN_CASH_BUFFER:.0f})')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate
    if args.min_dte < 0 or args.max_dte > 365:
        print("ERROR: DTE must be between 0 and 365")
        sys.exit(1)
    
    if args.min_pop < 0 or args.min_pop > 100:
        print("ERROR: Min POP must be between 0 and 100")
        sys.exit(1)
    
    if args.max_loss > args.account:
        print(f"ERROR: Max loss cannot exceed account total (${args.account:.0f})")
        sys.exit(1)
    
    # Run scan
    scanner = QuantScanner(account_total=args.account,
                           max_risk_per_trade=args.max_risk,
                           min_cash_buffer=args.min_cash)
    results = scanner.scan_multiple(
        tickers=args.tickers,
        mode=args.mode,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        max_loss_limit=args.max_loss,
        min_pop=args.min_pop / 100.0,  # Convert % to decimal
        min_width=float(args.min_width),
        max_width=float(args.max_width),
        json_output=args.json
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if not results:
        sys.exit(1)
    
    # Check if any executable trades found
    executable = any(
        any(s.get('fits_account') and s.get('pop', 0) > 0.6 
            for s in r.get('strategies', []))
        for r in results
    )
    
    sys.exit(0 if executable else 1)


if __name__ == '__main__':
    main()
