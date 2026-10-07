// Black–Scholes maths, mirroring src/tasks.py so lessons compute the same numbers
// as the experiments. Works in the browser (window.BS) and in Node (require).

(function (root) {
  const RATE = 0.02; // BS_RATE
  const M_SCALE = 0.2; // BS_M_SCALE
  const T_MIN = 0.1;
  const T_MAX = 2.0;

  // Standard normal CDF via erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)
  function erf(x) {
    const s = Math.sign(x);
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
        0.254829592) *
        t *
        Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

  // bs_call_over_spot: call price divided by spot, with K/S = exp(m)
  function callOverSpot(m, T, sigma, r = RATE) {
    const volSqrtT = sigma * Math.sqrt(T);
    const d1 = (-m + (r + 0.5 * sigma * sigma) * T) / volSqrtT;
    const d2 = d1 - volSqrtT;
    return normCdf(d1) - Math.exp(m - r * T) * normCdf(d2);
  }

  // Textbook form with explicit S and K, for comparison
  function call(S, K, T, sigma, r = RATE) {
    return S * callOverSpot(Math.log(K / S), T, sigma, r);
  }

  // bs_features: Gaussian input x = (x0, x1) -> (m, T)
  function features(x0, x1) {
    return { m: M_SCALE * x0, T: T_MIN + (T_MAX - T_MIN) * normCdf(x1) };
  }

  // Implied volatility by bisection (price is increasing in sigma)
  function impliedVol(price, m, T, r = RATE, lo = 0.01, hi = 2.0) {
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi);
      if (callOverSpot(m, T, mid, r) < price) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  const BS = { RATE, M_SCALE, T_MIN, T_MAX, normCdf, callOverSpot, call, features, impliedVol };
  if (typeof module !== "undefined" && module.exports) module.exports = BS;
  else root.BS = BS;
})(typeof window !== "undefined" ? window : globalThis);
