#include "WalkForwardEngine.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>

// ─── runTestWindow ────────────────────────────────────────────────────────────
// Core inner loop: run signal generation + trade execution on TEST bars.
// The KalmanFilter is passed in ALREADY WARMED UP on training data.
// We continue updating it on test bars (it keeps adapting, but signals
// are generated only from price data the model hasn't been fitted on).
std::vector<Trade> WalkForwardEngine::runTestWindow(
    const std::string& tickerA,
    const std::string& tickerB,
    const std::vector<double>& priceA,
    const std::vector<double>& priceB,
    KalmanFilter& kf,
    int testStart, int testEnd,
    double /*trainedBeta*/,
    double /*trainedAlpha*/)
{
    SignalGenerator sigGen(cfg_.signalCfg);
    PositionSizer   sizer(cfg_.sizingCfg);
    RiskManager     risk(cfg_.riskCfg);

    std::vector<Trade> trades;
    std::vector<double> spreadHistory;

    // State variables
    int    posSignal    = 0;       // +1, -1, or 0
    int    entryBar     = -1;
    double entrySpread  = 0.0;
    double entryZ       = 0.0;
    double entryBeta    = 0.0;
    double entryPriceA  = 0.0;     // price of A at entry - P&L must use THIS, not the exit-time price
    double positionSize = 0.0;
    TradeDirection dir  = TradeDirection::FLAT;

    // Running P&L for drawdown tracking (equity-based, so a pair that has
    // never had a winning trade still shows a real drawdown from initialNAV)
    double runningPnl    = 0.0;
    double peakEquity    = cfg_.initialNAV;
    int    holdBars      = 0;
    std::vector<double> recentTradePnls;   // trailing history feeding the VaR gate

    for (int i = testStart; i < testEnd; ++i) {
        if (i >= (int)priceA.size() || i >= (int)priceB.size()) break;

        double pA = priceA[i];
        double pB = priceB[i];

        // Update Kalman filter (continues adapting on test data)
        kf.update(pA, pB);
        double beta  = kf.getState().beta;
        double alpha = kf.getState().alpha;

        // Compute current spread
        double spread = pA - beta * pB - alpha;
        spreadHistory.push_back(spread);

        // Get signal for this bar
        auto bs = sigGen.updateBar(spread, spreadHistory, posSignal);

        // ── Handle open position ──
        if (posSignal != 0) {
            ++holdBars;

            bool forcedExit = (holdBars >= cfg_.signalCfg.maxHoldBars);
            bool naturalExit = (bs.exitSignal || bs.stopSignal)
                             && holdBars >= cfg_.signalCfg.minHoldBars;
            bool doExit = naturalExit || forcedExit;

            if (doExit) {
                // Close trade
                Trade t;
                t.tickerA       = tickerA;
                t.tickerB       = tickerB;
                t.entryBar      = entryBar;
                t.exitBar       = i;
                t.entrySpread   = entrySpread;
                t.exitSpread    = spread;
                t.entryZ        = entryZ;
                t.exitZ         = bs.zScore;
                t.hedgeRatio    = entryBeta;
                t.positionSize  = positionSize;
                t.direction     = dir;
                t.stoppedOut    = bs.stopSignal && naturalExit;

                // Gross P&L: for long spread (long A, short B×β)
                // P&L ≈ (exitSpread - entrySpread) × shares
                // Shares must be computed from the ENTRY price - the
                // number of shares bought at entry does not change just
                // because A's price has since moved.
                double spreadMove = spread - entrySpread;
                if (dir == TradeDirection::SHORT_A_LONG_B)
                    spreadMove = -spreadMove;

                double sharesA   = positionSize / (entryPriceA > 1e-6 ? entryPriceA : 1.0);
                t.grossPnl       = spreadMove * sharesA;
                t.transactionCost= risk.computeCost(positionSize).totalCostBps
                                   / 10000.0 * positionSize;
                t.pnl            = t.grossPnl - t.transactionCost;

                runningPnl += t.pnl;
                double curEquity = cfg_.initialNAV + runningPnl;
                if (curEquity > peakEquity) peakEquity = curEquity;

                trades.push_back(t);
                recentTradePnls.push_back(t.pnl);
                if (recentTradePnls.size() > 20) recentTradePnls.erase(recentTradePnls.begin());
                sizer.updateFromTrades(trades);

                posSignal = 0;
                holdBars  = 0;
            }
        }

        // ── Handle entry ──
        if (posSignal == 0 && bs.entrySignal) {
            double curEquity = cfg_.initialNAV + runningPnl;
            double pairDD    = (peakEquity - curEquity) / peakEquity;
            double portfolioDD = pairDD; // simplified single-pair (this engine trades one pair at a time)

            // Spread in bps: |spread| / avg_price * 10000
            double avgPrice  = (pA + pB) / 2.0;
            double spreadBps = std::abs(spread) / avgPrice * 10000.0;

            // openPositions is always 0 here: this loop only ever holds one
            // position at a time (posSignal==0 is the entry-branch guard),
            // so there is nothing else open to count against maxOpenPairs
            // within a single pair's test window.
            auto rc = risk.checkEntry(portfolioDD, pairDD,
                                       spreadBps, /*openPositions=*/0,
                                       cfg_.sizingCfg.portfolioNAV * 0.1,
                                       1e6, recentTradePnls);
            if (!rc.allowed()) {
                if (cfg_.verbose)
                    std::cout << "    [RISK BLOCK] bar=" << i
                              << " " << rc.reason << "\n";
                continue;
            }

            auto ps = sizer.compute(bs.zScore,
                                     cfg_.signalCfg.entryZScore,
                                     pA, pB, beta,
                                     sizer.winRateEstimate(),
                                     sizer.avgWinEstimate(),
                                     sizer.avgLossEstimate());

            posSignal    = bs.signal;
            entryBar     = i;
            entrySpread  = spread;
            entryZ       = bs.zScore;
            entryBeta    = beta;
            entryPriceA  = pA;
            positionSize = ps.notionalA;
            holdBars     = 0;
            dir = (bs.signal == +1)
                  ? TradeDirection::LONG_A_SHORT_B
                  : TradeDirection::SHORT_A_LONG_B;
        }
    }

    // Force-close any open position at end of window
    if (posSignal != 0 && entryBar >= 0) {
        int lastBar = std::min(testEnd - 1, (int)priceA.size() - 1);
        double spread = priceA[lastBar]
                      - kf.getState().beta * priceB[lastBar]
                      - kf.getState().alpha;

        Trade t;
        t.tickerA      = tickerA;
        t.tickerB      = tickerB;
        t.entryBar     = entryBar;
        t.exitBar      = lastBar;
        t.entrySpread  = entrySpread;
        t.exitSpread   = spread;
        t.entryZ       = entryZ;
        t.exitZ        = 0.0;
        t.hedgeRatio   = entryBeta;
        t.positionSize = positionSize;
        t.direction    = dir;
        t.stoppedOut   = false;

        double spreadMove = spread - entrySpread;
        if (dir == TradeDirection::SHORT_A_LONG_B) spreadMove = -spreadMove;
        double sharesA    = positionSize / (entryPriceA > 1e-6 ? entryPriceA : 1.0);
        t.grossPnl        = spreadMove * sharesA;
        t.transactionCost = risk.computeCost(positionSize).totalCostBps
                            / 10000.0 * positionSize;
        t.pnl             = t.grossPnl - t.transactionCost;
        trades.push_back(t);
    }

    return trades;
}

// ─── runSinglePair ────────────────────────────────────────────────────────────
TradeLog WalkForwardEngine::runSinglePair(
    const std::string& tickerA,
    const std::string& tickerB,
    const std::vector<double>& priceA,
    const std::vector<double>& priceB,
    bool verbose)
{
    int n = std::min(priceA.size(), priceB.size());
    if (n < cfg_.trainWindowBars + cfg_.testWindowBars) {
        std::cerr << "[WARN] Not enough bars for walk-forward on "
                  << tickerA << "/" << tickerB << "\n";
    }

    // Optimise Kalman delta on first training window
    double delta = 1e-4;
    if (cfg_.optimiseDelta && n > cfg_.trainWindowBars) {
        std::vector<double> trainA(priceA.begin(),
                                    priceA.begin() + cfg_.trainWindowBars);
        std::vector<double> trainB(priceB.begin(),
                                    priceB.begin() + cfg_.trainWindowBars);
        delta = KalmanFilter::optimiseDelta(trainA, trainB, verbose);
    }

    KalmanConfig kcfg;
    kcfg.delta = delta;
    KalmanFilter kf(kcfg);

    // Warm up on training data
    int warmEnd = std::min(cfg_.trainWindowBars, n);
    for (int i = 0; i < warmEnd; ++i)
        kf.update(priceA[i], priceB[i]);

    double trainBeta  = kf.getState().beta;
    double trainAlpha = kf.getState().alpha;

    // Run on full remaining data
    auto trades = runTestWindow(tickerA, tickerB, priceA, priceB,
                                 kf, warmEnd, n, trainBeta, trainAlpha);

    TradeLog log;
    for (auto& t : trades) log.addTrade(t);
    log.finalise(cfg_.initialNAV);

    if (verbose) log.printSummary();
    return log;
}

// ─── run (walk-forward) ───────────────────────────────────────────────────────
std::vector<WindowResult> WalkForwardEngine::run(
    const std::vector<std::string>& tickers,
    const std::unordered_map<std::string, std::vector<double>>& prices,
    TradeLog& combinedLog)
{
    // Find minimum series length
    int minLen = INT_MAX;
    for (auto& t : tickers) {
        auto it = prices.find(t);
        if (it != prices.end())
            minLen = std::min(minLen, (int)it->second.size());
    }
    if (minLen == INT_MAX || minLen < cfg_.trainWindowBars + cfg_.testWindowBars) {
        std::cerr << "[ERR] Insufficient data for walk-forward.\n";
        return {};
    }

    CointegrationEngine coint;
    std::vector<WindowResult> results;
    combinedLog.clear();

    int windowIdx = 0;
    int trainStart = 0;

    while (true) {
        int trainEnd  = trainStart + cfg_.trainWindowBars;
        int testStart = trainEnd;
        int testEnd   = testStart + cfg_.testWindowBars;

        if (testEnd > minLen) break;

        if (cfg_.verbose)
            std::cout << "\n══════ Walk-Forward Window " << windowIdx
                      << " [train: " << trainStart << "–" << trainEnd
                      << ", test: " << testStart << "–" << testEnd << "] ══════\n";

        // ── Extract training slices ──
        std::unordered_map<std::string, std::vector<double>> trainPrices;
        for (auto& t : tickers) {
            auto it = prices.find(t);
            if (it == prices.end()) continue;
            auto& v = it->second;
            trainPrices[t] = std::vector<double>(v.begin() + trainStart,
                                                   v.begin() + trainEnd);
        }

        // ── Screen pairs on training data ──
        auto candidates = coint.screenAllPairs(tickers, trainPrices,
                                                cfg_.verbose);
        if (candidates.empty()) {
            if (cfg_.verbose)
                std::cout << "  [SKIP] No cointegrated pair found in window "
                          << windowIdx << "\n";
            trainStart += cfg_.stepBars;
            ++windowIdx;
            continue;
        }

        // Use the best-ranked pair
        const PairCandidate& best = candidates[0];
        if (cfg_.verbose)
            std::cout << "  Selected pair: " << best.tickerA << "/"
                      << best.tickerB << "  β=" << best.beta
                      << "  HL=" << (int)best.halfLifeDays << "d\n";

        // ── Get full-length price series for this pair ──
        auto itA = prices.find(best.tickerA);
        auto itB = prices.find(best.tickerB);
        if (itA == prices.end() || itB == prices.end()) {
            trainStart += cfg_.stepBars;
            ++windowIdx;
            continue;
        }
        const auto& fullA = itA->second;
        const auto& fullB = itB->second;

        // ── Fit Kalman on training slice ──
        std::vector<double> trainA(fullA.begin() + trainStart,
                                    fullA.begin() + trainEnd);
        std::vector<double> trainB(fullB.begin() + trainStart,
                                    fullB.begin() + trainEnd);

        double delta = 1e-4;
        if (cfg_.optimiseDelta)
            delta = KalmanFilter::optimiseDelta(trainA, trainB, false);

        KalmanConfig kcfg; kcfg.delta = delta;
        KalmanFilter kf(kcfg);
        kf.warmUp(trainA, trainB);

        double trainBeta  = kf.getState().beta;
        double trainAlpha = kf.getState().alpha;

        // ── In-sample metrics (on training data) ──
        // Use a COPY of kf: it was already warmed up on exactly this bar
        // range, so re-running the same bars through the original kf
        // would update it on the training data a second time and leave
        // it in a state nobody else uses. A copy keeps the IS pass a
        // clean fit-then-evaluate rather than accidental double-adaptation.
        KalmanFilter kfIs = kf;
        auto isTrades = runTestWindow(best.tickerA, best.tickerB,
                                       fullA, fullB, kfIs,
                                       trainStart, trainEnd,
                                       trainBeta, trainAlpha);
        TradeLog isLog;
        for (auto& t : isTrades) isLog.addTrade(t);
        isLog.finalise(cfg_.initialNAV);

        // Re-warm for OOS (reset to training-end state)
        KalmanFilter kfOos(kcfg);
        kfOos.warmUp(trainA, trainB);

        // ── Out-of-sample run on test data ──
        auto oosTrades = runTestWindow(best.tickerA, best.tickerB,
                                        fullA, fullB, kfOos,
                                        testStart, testEnd,
                                        trainBeta, trainAlpha);
        TradeLog oosLog;
        for (auto& t : oosTrades) oosLog.addTrade(t);
        oosLog.finalise(cfg_.initialNAV);

        // Add OOS trades to combined log
        for (auto& t : oosTrades) combinedLog.addTrade(t);

        // ── Record window result ──
        WindowResult wr;
        wr.windowIdx    = windowIdx;
        wr.trainStart   = trainStart;
        wr.trainEnd     = trainEnd;
        wr.testStart    = testStart;
        wr.testEnd      = testEnd;
        wr.selectedPair = best;
        wr.kalmanDelta  = delta;
        wr.numTrades    = (int)oosTrades.size();
        wr.oosNetPnl    = oosLog.totalNetPnl();
        wr.oosSharpe    = oosLog.sharpe();
        wr.oosMaxDD     = oosLog.maxDD();
        wr.oosWinRate   = oosLog.winRate();
        wr.oosCalmar    = oosLog.calmar();
        wr.isSharpe     = isLog.sharpe();
        wr.isNetPnl     = isLog.totalNetPnl();

        results.push_back(wr);

        // ── Slide window ──
        if (cfg_.expandingWindow)
            trainStart = 0;              // expanding: always start from 0
        else
            trainStart += cfg_.stepBars; // rolling: slide forward

        ++windowIdx;
    }

    combinedLog.finalise(cfg_.initialNAV);
    if (cfg_.verbose) {
        std::cout << "\n══════ Walk-Forward Complete: "
                  << results.size() << " windows ══════\n";
        printWindowTable(results);
        combinedLog.printSummary();
    }
    return results;
}

// ─── printWindowTable ─────────────────────────────────────────────────────────
void WalkForwardEngine::printWindowTable(const std::vector<WindowResult>& res) {
    if (res.empty()) return;
    std::cout << "\n";
    std::cout << std::left
              << std::setw(4)  << "Win"
              << std::setw(14) << "Pair"
              << std::setw(8)  << "Trades"
              << std::setw(12) << "OOS P&L"
              << std::setw(10) << "OOS Sharpe"
              << std::setw(10) << "IS Sharpe"
              << std::setw(10) << "MaxDD"
              << std::setw(10) << "WinRate"
              << "\n";
    std::cout << repeatStr("─", 78) << "\n";
    for (const auto& w : res) {
        std::string pair = w.selectedPair.tickerA + "/" + w.selectedPair.tickerB;
        std::cout << std::left
                  << std::setw(4)  << w.windowIdx
                  << std::setw(14) << pair.substr(0, 13)
                  << std::setw(8)  << w.numTrades
                  << std::setw(12) << std::fixed << std::setprecision(0) << w.oosNetPnl
                  << std::setw(10) << std::setprecision(2) << w.oosSharpe
                  << std::setw(10) << w.isSharpe
                  << std::setw(10) << std::setprecision(1) << w.oosMaxDD*100 << "%"
                  << std::setw(10) << std::setprecision(1) << w.oosWinRate*100 << "%"
                  << "\n";
    }
    std::cout << repeatStr("─", 78) << "\n\n";
}

double WalkForwardEngine::computeDD(const std::vector<double>& pnlSeries,
                                     double nav) const {
    std::vector<double> eq(pnlSeries.size() + 1);
    eq[0] = nav;
    for (size_t i = 0; i < pnlSeries.size(); ++i)
        eq[i+1] = eq[i] + pnlSeries[i];
    return Stats::maxDrawdown(eq);
}