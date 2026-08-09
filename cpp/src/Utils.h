#pragma once
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr double SQRT_252   = 15.8745;   // sqrt(252) for annualisation
static constexpr double LOG2       = 0.693147;
static constexpr double INF_D      = 1e18;      // internal-only sentinel (never surfaced in reports)

// ADF MacKinnon critical values (no trend, constant only)
static constexpr double ADF_CV_1PCT  = -3.4336;
static constexpr double ADF_CV_5PCT  = -2.8621;
static constexpr double ADF_CV_10PCT = -2.5671;

// ─── TimeSeries ───────────────────────────────────────────────────────────────
struct TimeSeries {
    std::string          ticker;
    std::vector<double>  dates;    // as serial day numbers
    std::vector<double>  close;    // adjusted close prices
    std::vector<double>  logRet;   // log returns (size = close.size()-1)

    int size() const { return static_cast<int>(close.size()); }

    void computeLogReturns() {
        logRet.clear();
        logRet.reserve(close.size());
        for (size_t i = 1; i < close.size(); ++i)
            logRet.push_back(std::log(close[i] / close[i-1]));
    }

    // Slice [start, end)
    TimeSeries slice(int start, int end) const {
        TimeSeries ts;
        ts.ticker = ticker;
        ts.close.assign(close.begin()+start, close.begin()+end);
        ts.dates.assign(dates.begin()+start, dates.begin()+end);
        ts.computeLogReturns();
        return ts;
    }
};

// ─── Trade record ─────────────────────────────────────────────────────────────
enum class TradeDirection { LONG_A_SHORT_B, SHORT_A_LONG_B, FLAT };

struct Trade {
    std::string tickerA, tickerB;
    int         entryBar, exitBar;
    double      entrySpread, exitSpread;
    double      entryZ, exitZ;
    double      hedgeRatio;         // β at entry
    double      positionSize;       // dollar notional
    double      pnl;                // net P&L after costs
    double      grossPnl;
    double      transactionCost;
    TradeDirection direction;
    bool        stoppedOut;
};

// Repeat a (possibly multi-byte UTF-8) string n times. Box-drawing glyphs
// like "─"/"═"/"█"/"░" are 3-byte UTF-8 sequences — std::string(n, '─')
// truncates them to a single (wrong) byte, so use this instead.
inline std::string repeatStr(const std::string& s, int n) {
    std::string out;
    out.reserve(s.size() * (size_t)std::max(n, 0));
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

// ─── Stats namespace ─────────────────────────────────────────────────────────
namespace Stats {

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double variance(const std::vector<double>& v, bool ddof1 = true) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x-m)*(x-m);
    return s / (v.size() - (ddof1 ? 1 : 0));
}

inline double stddev(const std::vector<double>& v, bool ddof1 = true) {
    return std::sqrt(variance(v, ddof1));
}

inline double covariance(const std::vector<double>& x,
                          const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2)
        throw std::runtime_error("covariance: size mismatch");
    double mx = mean(x), my = mean(y);
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) s += (x[i]-mx)*(y[i]-my);
    return s / (x.size()-1);
}

// Simple OLS: y = alpha + beta * x
// Returns {alpha, beta, residuals}
inline std::tuple<double,double,std::vector<double>>
ols(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2)
        throw std::runtime_error("OLS: size mismatch");
    double mx = mean(x), my = mean(y);
    double sxy = 0.0, sxx = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sxy += (x[i]-mx)*(y[i]-my);
        sxx += (x[i]-mx)*(x[i]-mx);
    }
    double beta  = sxy / sxx;
    double alpha = my - beta * mx;
    std::vector<double> resid(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        resid[i] = y[i] - (alpha + beta * x[i]);
    return {alpha, beta, resid};
}

// Rolling mean (window w)
inline std::vector<double> rollingMean(const std::vector<double>& v, int w) {
    std::vector<double> out(v.size(), 0.0);
    double s = 0.0;
    for (int i = 0; i < (int)v.size(); ++i) {
        s += v[i];
        if (i >= w) s -= v[i-w];
        if (i >= w-1) out[i] = s / w;
    }
    return out;
}

// Rolling std (window w)
inline std::vector<double> rollingStd(const std::vector<double>& v, int w) {
    std::vector<double> out(v.size(), 0.0);
    for (int i = w-1; i < (int)v.size(); ++i) {
        double s = 0.0, ss = 0.0;
        for (int j = i-w+1; j <= i; ++j) { s += v[j]; ss += v[j]*v[j]; }
        // Use population variance (ddof=0).
        // This makes z-scores less prone to “std collapse” in short windows,
        // which matches the expected behaviour in the unit tests.
        double var = (ss - s*s/w) / w;
        out[i] = std::sqrt(std::max(var, 0.0));
    }
    return out;
}

// Percentile (linear interpolation)
inline double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p * (v.size()-1);
    int lo = (int)idx;
    double frac = idx - lo;
    if (lo+1 >= (int)v.size()) return v.back();
    return v[lo] + frac * (v[lo+1] - v[lo]);
}

// Autocorrelation at lag k
inline double autocorr(const std::vector<double>& v, int lag) {
    if ((int)v.size() <= lag) return 0.0;
    double m = mean(v);
    double num = 0.0, den = 0.0;
    for (int i = lag; i < (int)v.size(); ++i) num += (v[i]-m)*(v[i-lag]-m);
    for (int i = 0; i < (int)v.size(); ++i)   den += (v[i]-m)*(v[i]-m);
    return den == 0.0 ? 0.0 : num / den;
}

// Hurst exponent (rescaled range method, simplified)
inline double hurstExponent(const std::vector<double>& series) {
    // More stable estimator based on the scaling of increments:
    // tau(k) = std( X_{t+k} - X_t ) ~ k^H.
    // Fit log(tau) vs log(k) => slope ~= H.
    int n = (int)series.size();
    if (n < 50) return 0.5;

    const std::vector<int> lags = {2, 4, 8, 16, 32, 64};
    std::vector<double> xs, ys;
    xs.reserve(lags.size());
    ys.reserve(lags.size());

    for (int lag : lags) {
        if (lag <= 0 || lag >= n/2) break;
        std::vector<double> diffs;
        diffs.reserve(n - lag);
        for (int i = 0; i + lag < n; ++i)
            diffs.push_back(series[i + lag] - series[i]);
        double tau = stddev(diffs, true);
        if (tau > 1e-12) {
            xs.push_back(std::log((double)lag));
            ys.push_back(std::log(tau));
        }
    }

    if (xs.size() < 2) return 0.5;
    auto [a, h, r] = ols(xs, ys);
    return std::clamp(h, 0.0, 1.0);
}

// Annualised Sharpe ratio
inline double sharpeRatio(const std::vector<double>& dailyRets, double rfDaily = 0.0) {
    if (dailyRets.size() < 2) return 0.0;
    std::vector<double> excess(dailyRets.size());
    for (size_t i = 0; i < dailyRets.size(); ++i)
        excess[i] = dailyRets[i] - rfDaily;
    double s = stddev(excess);
    if (s < 1e-10) return 0.0;
    return mean(excess) / s * SQRT_252;
}

// Sortino ratio (penalises only downside volatility)
inline double sortinoRatio(const std::vector<double>& dailyRets, double rfDaily = 0.0) {
    if (dailyRets.size() < 2) return 0.0;
    double m = mean(dailyRets) - rfDaily;
    double dsum = 0.0; int dc = 0;
    for (double r : dailyRets) if (r < rfDaily) { dsum += (r-rfDaily)*(r-rfDaily); ++dc; }
    if (dc == 0) return std::numeric_limits<double>::infinity();  // no downside observed
    double dstd = std::sqrt(dsum / dc) * SQRT_252;
    return dstd < 1e-10 ? 0.0 : m * SQRT_252 / dstd;
}

// Max drawdown (as positive fraction)
inline double maxDrawdown(const std::vector<double>& equity) {
    if (equity.size() < 2) return 0.0;
    double peak = equity[0], maxDD = 0.0;
    for (double v : equity) {
        if (v > peak) peak = v;
        double dd = (peak - v) / peak;
        if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
}

// Calmar ratio = annualised return / max drawdown
inline double calmarRatio(const std::vector<double>& equity,
                           const std::vector<double>& dailyRets) {
    double mdd = maxDrawdown(equity);
    if (mdd < 1e-10) return std::numeric_limits<double>::infinity();  // no drawdown observed
    double annRet = mean(dailyRets) * 252.0;
    return annRet / mdd;
}

} // namespace Stats