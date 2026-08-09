#include "CointegrationEngine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>

// ─── MacKinnon p-value approximation ─────────────────────────────────────────
double CointegrationEngine::mackinnon_pvalue(double stat, int /*n*/) {
    // Piecewise-linear interpolation between known MacKinnon (1994/2010)
    // critical-value/p-value anchor points for the no-trend, constant-only
    // case. This is NOT the full MacKinnon response-surface polynomial —
    // it is an approximation good enough for pair screening/ranking, not
    // for publication-grade inference.
    // Anchors: t=-5.0→p=0.0001, t=-4.0→p=0.001, t=-3.45→p=0.01,
    //          t=-2.87→p=0.05,  t=-2.57→p=0.10, t=-1.94→p=0.25, t=0→p≈0.75
    double t = stat;
    if (t < -5.0) return 0.0001;
    if (t > 0.0)  return 0.9999;

    double p;
    if (t <= -4.0) {
        // [-5.0, 0.0001] to [-4.0, 0.001]
        p = 0.0001 + (t + 5.0) / 1.0 * 0.0009;
    } else if (t <= -3.45) {
        // [-4.0, 0.001] to [-3.45, 0.01]
        p = 0.001 + (t + 4.0) / 0.55 * 0.009;
    } else if (t <= -2.87) {
        // [-3.45, 0.01] to [-2.87, 0.05]
        p = 0.01 + (t + 3.45) / (3.45 - 2.87) * 0.04;
    } else if (t <= -2.57) {
        // [-2.87, 0.05] to [-2.57, 0.10]
        p = 0.05 + (t + 2.87) / (2.87 - 2.57) * 0.05;
    } else if (t <= -1.94) {
        // [-2.57, 0.10] to [-1.94, 0.25]
        p = 0.10 + (t + 2.57) / (2.57 - 1.94) * 0.15;
    } else {
        // above -1.94: increasingly likely to not reject
        p = 0.25 + (t + 1.94) / 1.94 * 0.50;
        p = std::min(p, 0.99);
    }
    return std::max(0.0001, std::min(0.9999, p));
}

// ─── Compute spread ──────────────────────────────────────────────────────────
std::vector<double> CointegrationEngine::computeSpread(
    const std::vector<double>& priceA,
    const std::vector<double>& priceB,
    double beta, double alpha)
{
    size_t n = std::min(priceA.size(), priceB.size());
    std::vector<double> spread(n);
    for (size_t i = 0; i < n; ++i)
        spread[i] = priceA[i] - beta * priceB[i] - alpha;
    return spread;
}

// ─── Half-life from OU regression ────────────────────────────────────────────
// Fit: ΔX_t = a + b * X_{t-1} + ε
// θ (mean-reversion speed) = -ln(1 + b) / Δt  ≈ -b (for small b)
// half_life = ln(2) / θ
double CointegrationEngine::estimateHalfLife(const std::vector<double>& spread) {
    int n = spread.size();
    if (n < 10) return 999.0;

    std::vector<double> delta(n-1), lagged(n-1);
    for (int i = 0; i < n-1; ++i) {
        delta[i]  = spread[i+1] - spread[i];
        lagged[i] = spread[i];
    }

    // OLS: Δspread = a + b * spread[t-1]
    auto [a, b, resid] = Stats::ols(lagged, delta);

    if (b >= 0.0) return 999.0;  // not mean-reverting
    double theta = -b;            // mean reversion speed (per bar)
    return LOG2 / theta;
}

// ─── Shared regression core (Frisch-Waugh partialling-out) ──────────────────
// Regress delta[t] on [const, level[t-1], delta[t-1], ..., delta[t-lags]]
// and return beta_hat on the level term, along with the true t-statistic
// and true residual sum of squares of that fit.
double CointegrationEngine::regressPartialled(const std::vector<double>& delta,
                                               const std::vector<double>& level,
                                               int lags, double& tStat, double& rss) const {
    int n = (int)delta.size();  // delta.size() == level.size()
    int T = n - lags;
    tStat = 0.0; rss = 0.0;
    int k = lags + 2;  // num parameters: const, level, lags×delta
    if (T <= k) return 0.0;

    std::vector<double> y_reg(T), x_level(T);
    std::vector<std::vector<double>> x_diff(lags, std::vector<double>(T));
    for (int i = 0; i < T; ++i) {
        y_reg[i]   = delta[i + lags];
        x_level[i] = level[i + lags];
        for (int l = 0; l < lags; ++l)
            x_diff[l][i] = delta[i + lags - 1 - l];
    }

    // Partial-out the constant and lag differences from y and x_level
    // using the Frisch-Waugh theorem (partial regression)
    auto partialOut = [&](const std::vector<double>& vec) {
        double m = Stats::mean(vec);
        std::vector<double> res(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) res[i] = vec[i] - m;
        for (int l = 0; l < lags; ++l) {
            double dot_xy = 0.0, dot_xx = 0.0;
            for (int i = 0; i < T; ++i) {
                dot_xy += res[i] * x_diff[l][i];
                dot_xx += x_diff[l][i] * x_diff[l][i];
            }
            if (dot_xx > 1e-12) {
                double coef = dot_xy / dot_xx;
                for (int i = 0; i < T; ++i) res[i] -= coef * x_diff[l][i];
            }
        }
        return res;
    };

    auto y_star = partialOut(y_reg);
    auto x_star = partialOut(x_level);

    double dot_xy = 0.0, dot_xx = 0.0;
    for (int i = 0; i < T; ++i) {
        dot_xy += y_star[i] * x_star[i];
        dot_xx += x_star[i] * x_star[i];
    }
    if (dot_xx < 1e-14) return 0.0;
    double beta_hat = dot_xy / dot_xx;

    for (int i = 0; i < T; ++i) {
        double e = y_star[i] - beta_hat * x_star[i];
        rss += e*e;
    }
    double s2 = rss / (T - k);
    double se = std::sqrt(std::max(s2, 0.0) / dot_xx);
    if (se >= 1e-12) tStat = beta_hat / se;
    return beta_hat;
}

// ─── Select ADF lag by BIC ───────────────────────────────────────────────────
// Fits the real ADF regression at each candidate lag (via the same
// partialling-out core used by runADFRegression) and picks the lag that
// minimises BIC on the true residual sum of squares.
int CointegrationEngine::selectADFLag(const std::vector<double>& delta,
                                       const std::vector<double>& lagged,
                                       int maxLags) const {
    int n = (int)delta.size();
    int bestLag = 0;
    double bestBIC = INF_D;

    for (int lag = 0; lag <= maxLags && lag < n/4; ++lag) {
        int T = n - lag;
        if (T < 15) break;
        int k = lag + 2;

        double tStat = 0.0, rss = 0.0;
        regressPartialled(delta, lagged, lag, tStat, rss);
        if (rss <= 0.0) continue;

        double bic = T * std::log(rss/T) + k * std::log(T);
        if (bic < bestBIC) { bestBIC = bic; bestLag = lag; }
    }
    return bestLag;
}

// ─── ADF Regression core ─────────────────────────────────────────────────────
// Returns ADF t-statistic (stored in tStat)
double CointegrationEngine::runADFRegression(const std::vector<double>& series,
                                              int lags,
                                              double& tStat) const {
    int n = (int)series.size();
    std::vector<double> delta(n-1), level(n-1);
    for (int i = 0; i < n-1; ++i) {
        delta[i] = series[i+1] - series[i];
        level[i] = series[i];
    }
    double rss = 0.0;
    return regressPartialled(delta, level, lags, tStat, rss);
}

// ─── ADF Test ────────────────────────────────────────────────────────────────
ADFResult CointegrationEngine::adfTest(const std::vector<double>& series,
                                        int maxLags) const {
    int n = series.size();
    if (n < 20) return {0.0, 1.0, 0, false, false};

    if (maxLags < 0) maxLags = (int)std::floor(12.0 * std::pow(n/100.0, 0.25));
    maxLags = std::min(maxLags, MAX_ADF_LAGS);

    // Build differences and lags for lag selection
    std::vector<double> delta(n-1), level(n-1);
    for (int i = 0; i < n-1; ++i) {
        delta[i] = series[i+1] - series[i];
        level[i] = series[i];
    }
    int bestLag = selectADFLag(delta, level, maxLags);

    double tStat = 0.0;
    runADFRegression(series, bestLag, tStat);

    double pval = mackinnon_pvalue(tStat, n);

    ADFResult res;
    res.statistic   = tStat;
    res.pValue      = pval;
    res.lags        = bestLag;
    res.rejectAt5Pct = (tStat < ADF_CV_5PCT);
    res.rejectAt1Pct = (tStat < ADF_CV_1PCT);
    return res;
}

// ─── Engle-Granger two-step cointegration test ───────────────────────────────
PairCandidate CointegrationEngine::engleGranger(
    const std::string& tickerA,
    const std::string& tickerB,
    const std::vector<double>& priceA,
    const std::vector<double>& priceB) const
{
    PairCandidate pc;
    pc.tickerA = tickerA;
    pc.tickerB = tickerB;
    pc.cointegrated = false;

    size_t n = std::min(priceA.size(), priceB.size());
    if (n < 50) return pc;

    std::vector<double> A(priceA.begin(), priceA.begin()+n);
    std::vector<double> B(priceB.begin(), priceB.begin()+n);

    // Step 1: OLS regression of A on B
    auto [alpha, beta, residuals] = Stats::ols(B, A);
    pc.alpha = alpha;
    pc.beta  = beta;

    // Reject negative or near-zero hedge ratios
    if (beta <= 0.05) return pc;

    // Step 2: ADF test on residuals
    auto adf = adfTest(residuals);
    pc.adfStat = adf.statistic;
    pc.pValue  = adf.pValue;

    // Hurst exponent on spread
    pc.hurstExponent = Stats::hurstExponent(residuals);

    // Half-life
    pc.halfLifeDays = estimateHalfLife(residuals);

    // Spread statistics
    pc.spreadMean = Stats::mean(residuals);
    pc.spreadStd  = Stats::stddev(residuals);

    // Cointegration criteria:
    //   1. ADF p-value < threshold (default 0.05)
    //   2. Hurst exponent < 0.5 (mean-reverting)
    //   3. Half-life in reasonable range
    bool adfPass   = (adf.pValue < pValueThreshold);
    bool hurstPass = (pc.hurstExponent < hurstThreshold);
    bool hlPass    = (pc.halfLifeDays >= minHalfLifeDays &&
                      pc.halfLifeDays <= maxHalfLifeDays);

    pc.cointegrated = adfPass && hurstPass && hlPass;

    // Score: lower is better (combine p-value and Hurst)
    pc.score = pc.pValue * 0.6 + pc.hurstExponent * 0.4;

    return pc;
}

// ─── Screen all pairs ────────────────────────────────────────────────────────
std::vector<PairCandidate>
CointegrationEngine::screenAllPairs(
    const std::vector<std::string>& tickers,
    const std::unordered_map<std::string, std::vector<double>>& prices,
    bool verbose) const
{
    int total = 0;
    int passed = 0;
    std::vector<PairCandidate> results;
    int n = tickers.size();
    int totalPairs = n * (n-1) / 2;

    if (verbose)
        std::cout << "[INFO] Screening " << totalPairs
                  << " pairs from " << n << " tickers...\n";

    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            const std::string& a = tickers[i];
            const std::string& b = tickers[j];

            auto itA = prices.find(a), itB = prices.find(b);
            if (itA == prices.end() || itB == prices.end()) continue;

            ++total;
            auto pc = engleGranger(a, b, itA->second, itB->second);

            if (pc.cointegrated) {
                ++passed;
                results.push_back(pc);
                if (verbose)
                    std::cout << "  [PASS] " << a << "/" << b
                              << "  β=" << std::fixed << std::setprecision(4) << pc.beta
                              << "  ADF=" << pc.adfStat
                              << "  p=" << pc.pValue
                              << "  H=" << pc.hurstExponent
                              << "  HL=" << (int)pc.halfLifeDays << "d\n";
            }
        }
    }

    // Sort by score (ascending = better)
    std::sort(results.begin(), results.end(),
              [](const PairCandidate& a, const PairCandidate& b){
                  return a.score < b.score;
              });

    if (verbose)
        std::cout << "[INFO] Cointegration screening: " << passed
                  << "/" << total << " pairs passed.\n\n";

    return results;
}