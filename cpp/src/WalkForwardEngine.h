#pragma once
#include "Utils.h"
#include "CointegrationEngine.h"
#include "KalmanFilter.h"
#include "SignalGenerator.h"
#include "PositionSizer.h"
#include "RiskManager.h"
#include "TradeLog.h"

// ─── WindowResult ─────────────────────────────────────────────────────────────
// Holds in-sample and out-of-sample metrics for one walk-forward window.
struct WindowResult {
    int    windowIdx;
    int    trainStart, trainEnd;   // bar indices
    int    testStart,  testEnd;

    // Pair selected in this window
    PairCandidate selectedPair;
    double        kalmanDelta;     // optimised delta for this window

    // Out-of-sample metrics (what actually matters)
    int    numTrades;
    double oosNetPnl;
    double oosSharpe;
    double oosMaxDD;
    double oosWinRate;
    double oosCalmar;

    // In-sample metrics (for overfitting comparison)
    double isSharpe;
    double isNetPnl;
};

// ─── BacktestConfig ───────────────────────────────────────────────────────────
struct BacktestConfig {
    int    trainWindowBars   = 252;  // 1 year training
    int    testWindowBars    = 63;   // 1 quarter test
    int    stepBars          = 63;   // slide by 1 quarter
    bool   expandingWindow   = false;// false = rolling, true = expanding
    bool   optimiseDelta     = true; // Kalman delta grid search
    bool   verbose           = true;
    double initialNAV        = 1'000'000.0;

    SignalConfig  signalCfg;
    SizingConfig  sizingCfg;
    RiskConfig    riskCfg;
};

// ─── WalkForwardEngine ────────────────────────────────────────────────────────
// Implements walk-forward backtesting to eliminate look-ahead bias:
//
//  Train window │ Test window │ (slide by stepBars)
//  ─────────────┼─────────────┼──────────────────────
//  fit coint.   │ run signals │
//  fit Kalman   │ trade       │
//  fit OU       │ compute P&L │
//
// Each window independently:
//   1. Screens all pairs on TRAIN data only
//   2. Selects best cointegrated pair
//   3. Fits Kalman filter on TRAIN data
//   4. Generates signals on TEST data
//   5. Executes trades on TEST data
//   6. Records OOS P&L
//
// Final equity curve is concatenation of all OOS windows.
// ─────────────────────────────────────────────────────────────────────────────
class WalkForwardEngine {
public:
    explicit WalkForwardEngine(const BacktestConfig& cfg = BacktestConfig{})
        : cfg_(cfg) {}

    // Run walk-forward backtest on a universe of price series
    // Returns per-window results + full combined trade log
    std::vector<WindowResult> run(
        const std::vector<std::string>& tickers,
        const std::unordered_map<std::string, std::vector<double>>& prices,
        TradeLog& combinedLog);

    // Run a single full-period backtest on a given pair (no walk-forward)
    // Useful for pair-level analysis after screening
    TradeLog runSinglePair(const std::string& tickerA,
                            const std::string& tickerB,
                            const std::vector<double>& priceA,
                            const std::vector<double>& priceB,
                            bool verbose = true);

    // Print walk-forward summary table
    static void printWindowTable(const std::vector<WindowResult>& results);

    const BacktestConfig& config() const { return cfg_; }

private:
    BacktestConfig cfg_;

    // Run one test window given fitted Kalman + signal config
    // Returns trades generated during [testStart, testEnd)
    std::vector<Trade> runTestWindow(
        const std::string& tickerA,
        const std::string& tickerB,
        const std::vector<double>& priceA,
        const std::vector<double>& priceB,
        KalmanFilter& kf,
        int testStart, int testEnd,
        double trainedBeta,
        double trainedAlpha);

    // Compute drawdown from a P&L series starting at initialNAV
    double computeDD(const std::vector<double>& pnlSeries, double nav) const;
};