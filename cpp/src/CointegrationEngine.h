#pragma once
#include "Utils.h"

// ─── ADF Result ───────────────────────────────────────────────────────────────
struct ADFResult {
    double statistic;       // ADF test statistic (more negative = more evidence of stationarity)
    double pValue;          // approximate MacKinnon p-value
    int    lags;            // lags used
    bool   rejectAt5Pct;    // true if we reject unit root at 5%
    bool   rejectAt1Pct;
};

// ─── PairCandidate ────────────────────────────────────────────────────────────
struct PairCandidate {
    std::string tickerA, tickerB;
    double      beta;           // OLS hedge ratio (initial, static)
    double      alpha;          // OLS intercept
    double      adfStat;        // ADF on spread residuals
    double      pValue;
    double      hurstExponent;  // < 0.5 = mean-reverting
    double      halfLifeDays;   // from OU model
    double      spreadMean;
    double      spreadStd;
    bool        cointegrated;   // passes both ADF + Hurst filter
    double      score;          // ranking score (lower p-value + lower Hurst = better)
};

// ─── CointegrationEngine ─────────────────────────────────────────────────────
// Implements:
//   1. Augmented Dickey-Fuller (ADF) test on spread residuals
//   2. Engle-Granger two-step cointegration test
//   3. Hurst exponent filter (H < 0.5 confirms mean reversion)
//   4. Half-life estimation from discrete OU regression
//   5. Batch screening of all pairs from a universe
// ─────────────────────────────────────────────────────────────────────────────
class CointegrationEngine {
public:
    // Maximum lags to try in ADF (BIC-selected)
    static constexpr int MAX_ADF_LAGS = 10;

    // ADF critical p-value threshold
    double pValueThreshold = 0.05;

    // Hurst threshold (series must be mean-reverting)
    // Use a slightly relaxed default; finite-sample H estimates can be noisy.
    double hurstThreshold = 0.6;

    // Minimum half-life in days (reject trivially fast mean-reverters)
    double minHalfLifeDays = 0.0;

    // Maximum half-life in days (reject non-reverting pairs)
    double maxHalfLifeDays = 120.0;

    // Run ADF test on a single series
    ADFResult adfTest(const std::vector<double>& series, int maxLags = -1) const;

    // Full Engle-Granger test on two price series
    // Returns the best PairCandidate (or cointegrated=false if fails)
    PairCandidate engleGranger(const std::string& tickerA,
                                const std::string& tickerB,
                                const std::vector<double>& priceA,
                                const std::vector<double>& priceB) const;

    // Screen all pairs from a universe
    // Returns sorted (by score) list of cointegrated pairs
    std::vector<PairCandidate>
    screenAllPairs(const std::vector<std::string>& tickers,
                   const std::unordered_map<std::string, std::vector<double>>& prices,
                   bool verbose = true) const;

    // Compute spread: spread[i] = priceA[i] - beta * priceB[i] - alpha
    static std::vector<double> computeSpread(const std::vector<double>& priceA,
                                              const std::vector<double>& priceB,
                                              double beta, double alpha);

    // Estimate half-life from OU regression on spread
    static double estimateHalfLife(const std::vector<double>& spread);

    // Approximate MacKinnon p-value from ADF statistic
    // (polynomial approximation for the no-trend case)
    static double mackinnon_pvalue(double adfStat, int n);

private:
    // Select optimal ADF lag by BIC (fits the real regression at each
    // candidate lag and compares true residual sum of squares)
    int selectADFLag(const std::vector<double>& diff,
                     const std::vector<double>& lagged,
                     int maxLags) const;

    // OLS for ADF regression:
    // Δy_t = α + β*y_{t-1} + γ_1*Δy_{t-1} + ... + γ_p*Δy_{t-p} + ε
    double runADFRegression(const std::vector<double>& series, int lags,
                             double& tStat) const;

    // Shared regression core (Frisch-Waugh partialling-out) operating
    // directly on pre-built delta/level arrays. Returns beta_hat and
    // fills tStat + rss (true residual sum of squares of the fit).
    double regressPartialled(const std::vector<double>& delta,
                              const std::vector<double>& level,
                              int lags, double& tStat, double& rss) const;
};