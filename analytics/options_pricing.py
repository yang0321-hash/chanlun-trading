"""
期权定价与Greeks模块

支持:
  - Black-Scholes 欧式期权定价
  - Greeks: Delta, Gamma, Theta, Vega, Rho
  - 隐含波动率 (Brentq)
  - 期权链分析
  - 市场情绪指标: PCR, IV-Based sentiment
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy import stats
from scipy.optimize import brentq


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def to_dict(self) -> dict:
        return {'Delta': f'{self.delta:.4f}', 'Gamma': f'{self.gamma:.4f}',
                'Theta': f'{self.theta:.4f}', 'Vega': f'{self.vega:.4f}',
                'Rho': f'{self.rho:.4f}'}


@dataclass
class OptionPrice:
    call_price: float
    put_price: float
    greeks_call: Greeks
    greeks_put: Greeks
    implied_vol: float
    strike: float
    spot: float
    expiry_days: int

    def to_dict(self) -> dict:
        return {
            'spot': f'{self.spot:.2f}', 'strike': f'{self.strike:.2f}',
            'call': f'{self.call_price:.4f}', 'put': f'{self.put_price:.4f}',
            'IV': f'{self.implied_vol:.2%}',
            'call_greeks': self.greeks_call.to_dict(),
            'put_greeks': self.greeks_put.to_dict(),
        }

    def summary(self) -> str:
        return (f'K={self.strike:.0f} T={self.expiry_days}d IV={self.implied_vol:.1%} | '
                f'Call={self.call_price:.4f} Put={self.put_price:.4f} | '
                f'Delta(C)={self.greeks_call.delta:.3f}')


@dataclass
class MarketSentiment:
    pcr_volume: float
    pcr_oi: float
    avg_iv_call: float
    avg_iv_put: float
    iv_skew: float
    sentiment: str
    confidence: float

    def to_dict(self) -> dict:
        return {'PCR': f'{self.pcr_volume:.2f}', 'IV_skew': f'{self.iv_skew:.2%}',
                'sentiment': self.sentiment, 'confidence': f'{self.confidence:.0%}'}

    def summary(self) -> str:
        return f'情绪: {self.sentiment} ({self.confidence:.0%}) | PCR={self.pcr_volume:.2f}'


class OptionsPricer:
    """期权定价器"""

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _d1_d2(spot, strike, T, r, sigma):
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return d1, d1 - sigma * np.sqrt(T)

    def bs_call(self, spot, strike, T, sigma, r=None):
        r = r or self.risk_free_rate
        if T <= 0 or sigma <= 0:
            return max(0, spot - strike)
        d1, d2 = self._d1_d2(spot, strike, T, r, sigma)
        return spot * stats.norm.cdf(d1) - strike * np.exp(-r * T) * stats.norm.cdf(d2)

    def bs_put(self, spot, strike, T, sigma, r=None):
        r = r or self.risk_free_rate
        if T <= 0 or sigma <= 0:
            return max(0, strike - spot)
        d1, d2 = self._d1_d2(spot, strike, T, r, sigma)
        return strike * np.exp(-r * T) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)

    def calc_greeks(self, spot, strike, T, sigma, option_type='call', r=None):
        r = r or self.risk_free_rate
        if T <= 0 or sigma <= 0:
            return Greeks(0, 0, 0, 0, 0)
        d1, d2 = self._d1_d2(spot, strike, T, r, sigma)
        sqrt_T = np.sqrt(T)
        pdf_d1 = stats.norm.pdf(d1)

        if option_type == 'call':
            delta = stats.norm.cdf(d1)
            theta = (-spot * pdf_d1 * sigma / (2 * sqrt_T)
                     - r * strike * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            delta = stats.norm.cdf(d1) - 1
            theta = (-spot * pdf_d1 * sigma / (2 * sqrt_T)
                     + r * strike * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365

        gamma = pdf_d1 / (spot * sigma * sqrt_T)
        vega = spot * pdf_d1 * sqrt_T / 100
        rho = (strike * T * np.exp(-r * T) *
               stats.norm.cdf(d2 if option_type == 'call' else -d2)) / 100
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    def implied_volatility(self, market_price, spot, strike, T, option_type='call', r=None):
        r = r or self.risk_free_rate
        intrinsic = max(0, spot - strike) if option_type == 'call' else max(0, strike - spot)
        if market_price <= intrinsic:
            return 0.0
        try:
            bs_func = self.bs_call if option_type == 'call' else self.bs_put
            return brentq(lambda sigma: bs_func(spot, strike, T, sigma, r) - market_price,
                          0.001, 5.0, xtol=1e-8)
        except (ValueError, RuntimeError):
            return np.nan

    def price_option(self, spot, strike, expiry_days, sigma=0.25, r=None) -> OptionPrice:
        r = r or self.risk_free_rate
        T = expiry_days / 365.0
        return OptionPrice(
            call_price=self.bs_call(spot, strike, T, sigma, r),
            put_price=self.bs_put(spot, strike, T, sigma, r),
            greeks_call=self.calc_greeks(spot, strike, T, sigma, 'call', r),
            greeks_put=self.calc_greeks(spot, strike, T, sigma, 'put', r),
            implied_vol=sigma, strike=strike, spot=spot, expiry_days=expiry_days,
        )

    def option_chain(self, spot, strikes, expiry_days, sigma=0.25) -> pd.DataFrame:
        rows = []
        for k in strikes:
            opt = self.price_option(spot, k, expiry_days, sigma)
            rows.append({
                'strike': k, 'moneyness': f'{spot/k:.2f}',
                'call': opt.call_price, 'put': opt.put_price,
                'call_delta': opt.greeks_call.delta, 'put_delta': opt.greeks_put.delta,
                'gamma': opt.greeks_call.gamma, 'vega': opt.greeks_call.vega,
            })
        return pd.DataFrame(rows)

    def market_sentiment(self, pcr_volume, pcr_oi, avg_iv_call, avg_iv_put, iv_skew) -> MarketSentiment:
        score = 0.5
        if pcr_volume > 1.5:
            score -= 0.2
        elif pcr_volume < 0.5:
            score += 0.2
        if iv_skew > 0.10:
            score -= 0.15
        elif iv_skew < -0.05:
            score += 0.08
        avg_iv = (avg_iv_call + avg_iv_put) / 2
        if avg_iv > 0.40:
            score -= 0.1
        score = max(0, min(1, score))
        if score > 0.65:
            sentiment = 'bullish'
        elif score > 0.45:
            sentiment = 'neutral'
        elif score > 0.25:
            sentiment = 'bearish'
        else:
            sentiment = 'extreme_fear'
        return MarketSentiment(
            pcr_volume=pcr_volume, pcr_oi=pcr_oi,
            avg_iv_call=avg_iv_call, avg_iv_put=avg_iv_put,
            iv_skew=iv_skew, sentiment=sentiment, confidence=abs(score - 0.5) * 2,
        )
