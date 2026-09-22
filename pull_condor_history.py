"""
Builds an SPY weekly $6/$6 iron condor backtest dataset for an arbitrary
date range, using the same methodology as the existing 2026 dataset:

  1. Pull SPY weekly OHLC bars for the range (get_equity_historicals,
     interval="week").
  2. Pull VIX weekly closes for the same range (get_index_historicals,
     instrument_id "3b912aa2-88f9-4682-8ae3-e39520bdf4db", interval="week").
  3. For each week's Monday (or first trading day if Monday is a
     holiday), estimate the short-put (~delta -0.16) and short-call
     (~delta +0.21) strikes using a Black-Scholes approximation seeded
     by that week's VIX level -- see estimate_strikes() below. This is
     the same approximation used to build the original dataset's
     later weeks, and it isn't exact: Robinhood's own reported IVs run
     a bit rich relative to real mid prices, and the vol split between
     puts/calls (put IV usually running a few points above call IV)
     was calibrated on 2026 data. Treat the resulting strikes as
     "close enough for backtest purposes," not as what a live trader
     would have selected in the moment.
  4. Wings are the short strike +/- 6.
  5. For each of the 4 legs, resolve the real EXPIRED option contract
     (get_option_instruments, state="expired") and pull its hourly
     price history (get_option_historicals) from the Monday entry
     through expiry Friday.
  6. Record everything in the same JSON schema as
     spy_condor_current_strategy.json so it merges cleanly.

IMPORTANT CAVEAT for extreme-volatility periods (e.g. the COVID crash):
VIX spiked well outside the range this approximation was calibrated on
(SPY weekly VIX ~15-35 in the 2026 dataset; COVID-era VIX reached the
70s-80s). The strike-selection formula may put strikes further from
target deltas than usual in those weeks. This is a known limitation --
flag it in the output notes rather than silently trusting it. If you
have a way to pull actual historical greeks/IV for expired contracts,
prefer that over the estimate.

This script issues no trades and reads only expired/historical data.

Usage: edit RANGE_START, RANGE_END, OUTPUT_PATH below, then run twice
(once per historical window requested), or pass them as CLI args:

    python pull_condor_history.py 2025-01-06 2026-01-09 spy_condor_2025.json
    python pull_condor_history.py 2019-12-02 2020-12-28 spy_condor_covid.json

Claude Code should read this file, then use its own Robinhood MCP tool
calls to implement each TODO-marked step -- the functions below define
the exact logic and output shape, but the actual MCP tool invocations
need to be made by Code directly (this script can't call MCP tools
itself when run as a plain Python script; treat the function bodies
marked "# MCP CALL HERE" as pseudocode specifying what to call and
how to use the result).
"""

import sys
import json
import math
from datetime import datetime, timedelta

CHAIN_ID = "c277b118-58d9-4060-8dc5-a3b5898955cb"  # SPY option chain
VIX_INSTRUMENT_ID = "3b912aa2-88f9-4682-8ae3-e39520bdf4db"

SHORT_PUT_DELTA_TARGET = 0.16
SHORT_CALL_DELTA_TARGET = 0.21
WING_WIDTH = 6
SLIPPAGE_PER_LEG = 0.03

# Calibrated vol-split coefficients (put IV and call IV as a fraction
# of VIX) -- from the original dataset's strike-selection calibration.
# These may not hold in extreme regimes; sanity-check results.
PUT_VOL_COEF = 0.91
CALL_VOL_COEF = 0.60


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(S, K, T, sigma, is_call):
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1


def estimate_strikes(spy_open, vix, days_to_expiry):
    """
    Returns (short_put_strike, long_put_strike, short_call_strike, long_call_strike),
    all rounded to the nearest whole dollar (SPY weekly strikes are $1-wide).
    """
    T = (days_to_expiry + 6 / 6.5) / 252
    sigma_put = PUT_VOL_COEF * vix / 100
    sigma_call = CALL_VOL_COEF * vix / 100

    best_put, best_put_diff = None, 1e9
    for K in range(int(spy_open) - 80, int(spy_open)):
        d = abs(abs(bs_delta(spy_open, K, T, sigma_put, False)) - SHORT_PUT_DELTA_TARGET)
        if d < best_put_diff:
            best_put_diff, best_put = d, K

    best_call, best_call_diff = None, 1e9
    for K in range(int(spy_open) + 1, int(spy_open) + 80):
        d = abs(bs_delta(spy_open, K, T, sigma_call, True) - SHORT_CALL_DELTA_TARGET)
        if d < best_call_diff:
            best_call_diff, best_call = d, K

    return best_put, best_put - WING_WIDTH, best_call, best_call + WING_WIDTH


def monday_entries(start_date, end_date):
    """Yield each week's entry date (Monday, or first weekday if Monday
    is a holiday -- Code should verify against an actual market
    calendar; this just yields Mondays and skips known long weekends
    heuristically. Manual correction may be needed for a handful of
    weeks, same as was done for the 2026 dataset (e.g. Presidents Day,
    MLK Day weeks sometimes shifted the entry to Tuesday)."""
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    # advance to first Monday
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=7)


def week_expiry(entry_date_str):
    """Standard case: expiry is the Friday of the same week. Code
    should override for known short weeks (holiday-shortened, e.g.
    Good Friday) using the actual SPY weekly bar's last trading day."""
    d = datetime.strptime(entry_date_str, "%Y-%m-%d")
    friday = d + timedelta(days=(4 - d.weekday()))
    return friday.strftime("%Y-%m-%d")


def build_dataset(range_start, range_end, output_path):
    """
    High-level pipeline Code should implement, step by step:

    1. spy_weekly = MCP CALL HERE: get_equity_historicals(
           symbols=["SPY"], interval="week",
           start_time=f"{range_start}T00:00:00Z",
           end_time=f"{range_end}T00:00:00Z")
       -> gives open/high/low/close per week; use `open` as the
          Monday entry reference price for strike selection.

    2. vix_weekly = MCP CALL HERE: get_index_historicals(
           instrument_ids=[VIX_INSTRUMENT_ID], interval="week",
           start_time=..., end_time=...)
       -> gives that week's VIX level; use `close` of the PRIOR
          week (or the entry-day open, if available) as the vol
          input for that week's strike estimate -- don't use the
          current week's own close, since that wouldn't have been
          known at entry.

    3. For each week:
         a. spy_open, vix = pull from the series above
         b. days_to_expiry = trading days from entry to that week's
            Friday (usually 4; 3 for a Tue entry or Thu expiry)
         c. sp, lp, sc, lc = estimate_strikes(spy_open, vix, days_to_expiry)
         d. For each of the 4 (strike, type) pairs:
              instrument = MCP CALL HERE: get_option_instruments(
                  chain_id=CHAIN_ID, expiration_dates=[expiry_date],
                  state="expired", strike_price=f"{strike}.0000",
                  type="put" or "call")
              -> take instrument["data"]["instruments"][0]["id"]
         e. For each of the 4 instrument ids:
              bars = MCP CALL HERE: get_option_historicals(
                  instrument_ids=[id], interval="hour",
                  start_time=f"{entry_date}T14:00:00Z",  # adjust for
                                                            # DST -- see
                                                            # note below
                  end_time=f"{expiry_date}T22:00:00Z")
              -> entry_price = bars[0]["open_price"]
104              -> hourly_closes = [b["close_price"] for b in bars]
         f. settle = the SPY weekly bar's `close` price for that week
            (from step 1)
         g. Store all of this in the same shape as
            spy_condor_current_strategy.json's per-week entries.

    DST note: US clocks are 5 hours behind UTC (EST) from roughly
    early November to mid-March, and 4 hours behind (EDT) the rest of
    the year. 10:00 AM ET is therefore 15:00 UTC in EST months and
    14:00 UTC in EDT months. Get this wrong and every entry/exit in
    the affected weeks is pulled from the wrong hour.

    Skip/flag weeks where:
      - SPY had a holiday-shortened week (adjust days_to_expiry and
        expiry date accordingly -- check the weekly bar's date
        range or a market calendar)
      - An option instrument lookup returns no contract (can happen
        for very old, illiquid, or delisted strikes) -- note the week
        as "MISSING" rather than silently dropping it
      - The COVID week of 2020-03-16 saw circuit-breaker trading
        halts; hourly bars may be sparse or gapped that week --
        that's expected, not a data error.

    Write the final dict to output_path in this shape:

        {
          "schema_version": 1,
          "chain_id": CHAIN_ID,
          "chain_symbol": "SPY",
          "slippage_per_leg": SLIPPAGE_PER_LEG,
          "strategy": {... same as current-strategy export ...},
          "date_range": {"start": range_start, "end": range_end},
          "notes": "<mention the strike-estimation caveat, and list any
                     weeks that were skipped/missing/flagged>",
          "weeks": {
             "<label, e.g. 2025-01-06>": {
                "expiry": ..., "entry_date": ...,
                "settle": ...,
                "strikes": {"short_put":.., "long_put":.., "short_call":.., "long_call":..},
                "entry_prices": {"short_put":.., "long_put":.., "short_call":.., "long_call":..},
                "hourly_closes": {"short_put":[...], "long_put":[...], "short_call":[...], "long_call":[...]}
             },
             ...
          }
        }
    """
    raise NotImplementedError(
        "This function documents the pipeline for Code to implement using "
        "its own Robinhood MCP tool calls -- see the docstring above for "
        "the exact sequence, and estimate_strikes()/monday_entries()/"
        "week_expiry() above for the reusable pure-Python pieces."
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    range_start, range_end, output_path = sys.argv[1:4]
    build_dataset(range_start, range_end, output_path)
