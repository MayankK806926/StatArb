// tests/test_all.cpp
// ──────────────────────────────────────────────────────────────────────────────
// Self-contained unit test suite for StatArb components.
// No external framework - lightweight assertion macros only.
//
// Build:  g++ -std=c++17 -O2 test_all.cpp ../src/*.cpp -o test_statarb
// Run:    ./test_statarb
// ──────────────────────────────────────────────────────────────────────────────
#include "../src/Utils.h"
#include "../src/CointegrationEngine.h"
#include "../src/KalmanFilter.h"
#include "../src/SignalGenerator.h"
#include "../src/PositionSizer.h"
#include "../src/RiskManager.h"
#include "../src/TradeLog.h"
#include <iostream>
#include <functional>
#include <cassert>
#include <random>
#include <numeric>

// ─── Test harness ──────────────────────────────────────────────────────────
static int passed_ = 0, failed_ = 0;
static std::string cur_;

#define EXPECT_TRUE(e) do { if(!(e)) { \
    std::cerr<<"  [FAIL] "<<cur_<<"\n         "<<#e<<" was FALSE\n"; \
    ++failed_; return; } } while(0)
#define EXPECT_NEAR(a,b,tol) do { if(std::abs((a)-(b))>(tol)) { \
    std::cerr<<"  [FAIL] "<<cur_<<"\n         |"<<(a)<<" - "<<(b)<<"| > "<<(tol)<<"\n"; \
    ++failed_; return; } } while(0)
#define EXPECT_LT(a,b) EXPECT_TRUE((a)<(b))
#define EXPECT_GT(a,b) EXPECT_TRUE((a)>(b))
#define EXPECT_EQ(a,b) EXPECT_TRUE((a)==(b))

void runTest(const std::string& name, std::function<void()> fn) {
    cur_ = name;
    std::cout << "  [ RUN ] " << name << "\n";
    int failedBefore = failed_;
    fn();
    // Only mark the test as OK if it didn't introduce any new failures.
    if (failed_ == failedBefore) {
        std::cout << "  [ OK  ] " << name << "\n";
        ++passed_;
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
// Generate a synthetic cointegrated pair:
//   priceB = random walk
//   priceA = beta * priceB + alpha + stationary_noise
std::pair<std::vector<double>, std::vector<double>>
makeCointegratedPair(int n, double beta, double alpha,
                      double noiseVol, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0, 1);

    std::vector<double> B(n), A(n);
    double bPrice = 500.0;
    for (int i = 0; i < n; ++i) {
        bPrice *= std::exp(0.0001 + 0.015 * nd(rng));
        B[i] = bPrice;
        A[i] = beta * bPrice + alpha + noiseVol * nd(rng);
    }
    return {A, B};
}

// Generate a pure random walk (NOT cointegrated)
std::vector<double> makeRandomWalk(int n, double vol, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0, vol);
    std::vector<double> v(n);
    v[0] = 100.0;
    for (int i = 1; i < n; ++i) v[i] = v[i-1] * std::exp(nd(rng));
    return v;
}

// ─── Stats tests ──────────────────────────────────────────────────────────────
void test_stats_mean_std() {
    std::vector<double> v = {1,2,3,4,5};
    EXPECT_NEAR(Stats::mean(v), 3.0, 1e-10);
    EXPECT_NEAR(Stats::stddev(v), std::sqrt(2.5), 1e-6);
}

void test_stats_ols() {
    // y = 2x + 1 + tiny noise
    std::vector<double> x = {1,2,3,4,5};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};
    auto [alpha, beta, resid] = Stats::ols(x, y);
    EXPECT_NEAR(beta,  2.0, 1e-6);
    EXPECT_NEAR(alpha, 1.0, 1e-6);
    for (double r : resid) EXPECT_NEAR(r, 0.0, 1e-6);
}

void test_stats_sharpe() {
    // Sharpe of constant +1% daily returns = (0.01*252) / (0 * sqrt(252))
    // → infinite; use a finite case
    std::vector<double> rets(100, 0.005);
    // stddev is 0, Sharpe should return 0 (guard)
    double s = Stats::sharpeRatio(rets);
    EXPECT_NEAR(s, 0.0, 1e-6);  // zero std → returns 0

    // Realistic case
    std::vector<double> rets2(252);
    for (int i = 0; i < 252; ++i) rets2[i] = (i%2 == 0) ? 0.01 : -0.005;
    s = Stats::sharpeRatio(rets2);
    EXPECT_GT(s, 0.0);  // positive Sharpe since mean > 0
}

void test_stats_max_drawdown() {
    std::vector<double> eq = {100, 110, 105, 90, 95, 100};
    double dd = Stats::maxDrawdown(eq);
    // Peak 110, trough 90: DD = 20/110
    EXPECT_NEAR(dd, 20.0/110.0, 1e-6);
}

void test_hurst_mean_reverting() {
    // Generate OU-like mean-reverting series
    std::mt19937 rng(1);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> ou(500);
    ou[0] = 0.0;
    for (int i = 1; i < 500; ++i)
        ou[i] = 0.7 * ou[i-1] + 0.3 * nd(rng);  // strong mean reversion
    double H = Stats::hurstExponent(ou);
    EXPECT_LT(H, 0.5);  // should be < 0.5 for mean-reverting
}

void test_hurst_random_walk() {
    auto rw = makeRandomWalk(500, 0.01, 7);
    // Log-price series
    std::vector<double> logRW(rw.size());
    for (size_t i = 0; i < rw.size(); ++i) logRW[i] = std::log(rw[i]);
    double H = Stats::hurstExponent(logRW);
    // Random walk has H ≈ 0.5; give generous tolerance for small sample
    EXPECT_GT(H, 0.3);
    EXPECT_LT(H, 0.8);
}

// ─── CointegrationEngine tests ────────────────────────────────────────────────
void test_adf_rejects_stationary() {
    // AR(1) with coefficient 0.5 → stationary, ADF should reject unit root
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> ar(300);
    ar[0] = 0.0;
    for (int i = 1; i < 300; ++i) ar[i] = 0.4 * ar[i-1] + nd(rng);

    CointegrationEngine coint;
    auto res = coint.adfTest(ar);
    EXPECT_TRUE(res.rejectAt5Pct);   // must reject unit root (series is stationary)
}

void test_adf_cannot_reject_rw() {
    // Pure random walk → cannot reject unit root
    auto rw = makeRandomWalk(300, 0.01, 99);
    std::vector<double> logRW(rw.size());
    for (size_t i = 0; i < rw.size(); ++i) logRW[i] = std::log(rw[i]);

    CointegrationEngine coint;
    auto res = coint.adfTest(logRW);
    EXPECT_TRUE(!res.rejectAt1Pct); // should NOT strongly reject
}

void test_engle_granger_detects_coint() {
    auto [A, B] = makeCointegratedPair(500, 1.5, 20.0, 2.0, 42);
    CointegrationEngine coint;
    auto pc = coint.engleGranger("A", "B", A, B);
    EXPECT_TRUE(pc.cointegrated);
    EXPECT_GT(pc.beta, 0.5);
    EXPECT_LT(pc.pValue, 0.1);
}

void test_engle_granger_rejects_unrelated() {
    auto A = makeRandomWalk(400, 0.02, 1);
    auto B = makeRandomWalk(400, 0.02, 999);  // independent
    CointegrationEngine coint;
    auto pc = coint.engleGranger("A", "B", A, B);
    // Most of the time independent RWs won't pass strict cointegration test
    // We just check the function runs without error
    EXPECT_TRUE(pc.pValue >= 0.0 && pc.pValue <= 1.0);
}

void test_half_life_estimation() {
    // Generate OU process with known half-life ≈ 20 bars
    // θ = ln(2)/20 ≈ 0.0347 → AR coeff = exp(-θ) ≈ 0.966
    double theta = LOG2 / 20.0;
    double ar = std::exp(-theta);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> ou(500);
    ou[0] = 0.0;
    for (int i = 1; i < 500; ++i)
        ou[i] = ar * ou[i-1] + 0.5 * nd(rng);

    double hl = CointegrationEngine::estimateHalfLife(ou);
    // Allow wide tolerance: finite-sample estimation is noisy
    EXPECT_GT(hl, 5.0);
    EXPECT_LT(hl, 80.0);
}

// ─── KalmanFilter tests ───────────────────────────────────────────────────────
void test_kalman_tracks_beta() {
    // priceA = 1.5 * priceB + 10; Kalman should converge to β ≈ 1.5
    auto [A, B] = makeCointegratedPair(400, 1.5, 10.0, 1.0, 5);
    KalmanFilter kf;
    kf.init();
    for (size_t i = 0; i < A.size(); ++i) kf.update(A[i], B[i]);
    double beta = kf.getState().beta;
    EXPECT_NEAR(beta, 1.5, 0.3);  // within 0.3 of true β
}

void test_kalman_spread_near_zero() {
    // If A = 2*B exactly, spread should be near zero
    auto [A, B] = makeCointegratedPair(300, 2.0, 0.0, 0.1, 11);
    KalmanFilter kf;
    kf.init();
    kf.warmUp(A, B);
    // After warm-up, spread of last bar should be small
    double spread = kf.computeSpread(A.back(), B.back());
    EXPECT_LT(std::abs(spread), 20.0);  // generous: noise is 0.1 scale
}

void test_kalman_positive_covariance() {
    // P matrix diagonal should stay positive
    auto [A, B] = makeCointegratedPair(200, 1.0, 5.0, 2.0, 3);
    KalmanFilter kf; kf.init();
    for (size_t i = 0; i < A.size(); ++i) kf.update(A[i], B[i]);
    EXPECT_GT(kf.getState().P00, 0.0);
    EXPECT_GT(kf.getState().P11, 0.0);
}

void test_kalman_log_likelihood_increases_with_better_delta() {
    auto [A, B] = makeCointegratedPair(300, 1.5, 10.0, 1.5, 42);

    // Very large delta (too noisy) vs optimal delta
    KalmanConfig cfgLarge; cfgLarge.delta = 1.0;
    KalmanConfig cfgGood;  cfgGood.delta  = 1e-4;

    KalmanFilter kfL(cfgLarge), kfG(cfgGood);
    for (size_t i = 0; i < A.size(); ++i) {
        kfL.update(A[i], B[i]);
        kfG.update(A[i], B[i]);
    }
    // Good delta should give better (less negative) log-likelihood
    EXPECT_GT(kfG.getState().logLikelihood, kfL.getState().logLikelihood);
}

// ─── SignalGenerator tests ────────────────────────────────────────────────────
void test_signal_entry_at_2sigma() {
    // Build a spread that crosses +2σ
    std::vector<double> spread(60, 0.0);
    // Fill first 30 with near-zero (mean~0, std~1)
    std::mt19937 rng(1); std::normal_distribution<double> nd(0, 1);
    for (int i = 0; i < 30; ++i) spread[i] = nd(rng);
    // Bar 30: spike to +2.5σ above mean
    double m = Stats::mean(std::vector<double>(spread.begin(), spread.begin()+30));
    double s = Stats::stddev(std::vector<double>(spread.begin(), spread.begin()+30));
    spread[30] = m + 2.5 * s;

    SignalConfig cfg; cfg.zScoreWindow = 20; cfg.entryZScore = 2.0;
    SignalGenerator sg(cfg);
    auto signals = sg.generateSignals(spread);

    // Should get a short-spread entry near bar 30
    bool gotEntry = false;
    for (int i = 25; i < 35 && i < (int)signals.size(); ++i)
        if (signals[i].entrySignal) { gotEntry = true; break; }
    EXPECT_TRUE(gotEntry);
}

void test_signal_exit_at_zero() {
    // Build spread: starts near 0 (establishes rolling stats), jumps to +3,
    // then decays smoothly toward 0. A flat plateau before the decay was
    // tried originally but collapses the rolling std to ~0 inside the
    // window, which causes a premature spurious entry+exit before the
    // decay even starts - so the decay leg is continuous with the jump.
    std::vector<double> spread(100);
    for (int i = 0; i < 40; ++i) spread[i] = 0.0 + 0.1*(double)((i%5)-2);
    for (int i = 40; i < 100; ++i) spread[i] = 3.0 - (i-40) * 0.08;  // decay toward 0

    SignalConfig cfg; cfg.zScoreWindow = 20; cfg.entryZScore = 2.0;
    cfg.exitZScore = 0.5;
    SignalGenerator sg(cfg);
    auto sigs = sg.generateSignals(spread);

    bool gotEntry = false, gotExit = false;
    for (int i = 0; i < 100; ++i) {
        if (sigs[i].entrySignal) gotEntry = true;
        if (sigs[i].exitSignal)  gotExit  = true;
    }
    EXPECT_TRUE(gotEntry);
    EXPECT_TRUE(gotExit);
}

void test_signal_stop_at_3p5sigma() {
    std::vector<double> spread(80);
    for (int i = 0; i < 30; ++i) spread[i] = 0.1*(double)(i%3);
    // Enter at bar 30, spread keeps going → stop.
    // NOTE: a constant-slope ramp does NOT work here - the rolling window
    // adapts its own std to the ramp within a few bars, so a linear (or
    // even accelerating-but-smooth) divergence asymptotes to a fixed
    // z-score well below the stop threshold instead of blowing past it.
    // The divergence has to outrun the 20-bar window's ability to
    // renormalize, so jump sharply right after entry instead.
    double m = Stats::mean(std::vector<double>(spread.begin(), spread.begin()+30));
    double s = std::max(Stats::stddev(
        std::vector<double>(spread.begin(), spread.begin()+30)), 0.01);
    spread[30] = m + 2.2*s;
    for (int i = 31; i < 80; ++i)
        spread[i] = spread[30] + (i - 30) * s * 5.0;  // sharp jump right after entry → stop

    SignalConfig cfg; cfg.zScoreWindow = 20; cfg.entryZScore = 2.0;
    cfg.stopZScore = 3.5; cfg.exitZScore = 0.0;
    SignalGenerator sg(cfg);
    auto sigs = sg.generateSignals(spread);

    bool gotStop = false;
    for (auto& bs : sigs) if (bs.stopSignal) { gotStop = true; break; }
    EXPECT_TRUE(gotStop);
}

// ─── PositionSizer tests ──────────────────────────────────────────────────────
void test_position_size_positive() {
    SizingConfig cfg; cfg.portfolioNAV = 1'000'000;
    PositionSizer ps(cfg);
    auto sz = ps.compute(2.5, 2.0, 1000.0, 500.0, 1.5);
    EXPECT_GT(sz.notionalA, 0.0);
    EXPECT_GT(sz.notionalB, 0.0);
}

void test_position_size_capped() {
    SizingConfig cfg; cfg.portfolioNAV = 1'000'000; cfg.maxPairAllocation = 0.10;
    PositionSizer ps(cfg);
    auto sz = ps.compute(5.0, 2.0, 100.0, 50.0, 1.0, 0.9, 0.1, 0.001);
    // Should be capped at 10% of NAV = 100,000
    EXPECT_LT(sz.notionalA, 110'000.0);
}

void test_position_z_scaling() {
    SizingConfig cfg; cfg.portfolioNAV = 1'000'000;
    PositionSizer ps(cfg);
    auto sz_low  = ps.compute(2.1, 2.0, 100.0, 50.0, 1.0);
    auto sz_high = ps.compute(3.5, 2.0, 100.0, 50.0, 1.0);
    // Higher |z| should result in larger or equal position
    EXPECT_GT(sz_high.notionalA + 1.0, sz_low.notionalA);
}

// ─── RiskManager tests ────────────────────────────────────────────────────────
void test_risk_blocks_on_drawdown() {
    RiskConfig cfg; cfg.maxPortfolioDrawdown = 0.10;
    RiskManager rm(cfg);
    auto rc = rm.checkEntry(0.12, 0.0, 20.0, 1, 100'000);
    EXPECT_TRUE(!rc.allowed());
    EXPECT_EQ(rc.decision, RiskDecision::BLOCK_DRAWDOWN);
}

void test_risk_blocks_on_concentration() {
    RiskConfig cfg; cfg.maxOpenPairs = 3;
    RiskManager rm(cfg);
    auto rc = rm.checkEntry(0.0, 0.0, 20.0, 3, 100'000);
    EXPECT_TRUE(!rc.allowed());
    EXPECT_EQ(rc.decision, RiskDecision::BLOCK_CONCENTRATION);
}

void test_risk_allows_normal() {
    RiskConfig cfg;
    RiskManager rm(cfg);
    // spreadBps must clear the round-trip cost estimate (bid-ask + brokerage
    // + sqrt market-impact on this notional/adv ratio, ~52 bps here) or the
    // break-even check correctly blocks the trade - 80 bps clears it with room.
    auto rc = rm.checkEntry(0.02, 0.01, 80.0, 1, 50'000);
    EXPECT_TRUE(rc.allowed());
}

void test_var_calculation() {
    RiskManager rm;
    std::vector<double> pnl(100);
    std::iota(pnl.begin(), pnl.end(), -50.0);  // -50..49
    double var = rm.historicalVaR(pnl, 0.95);
    EXPECT_GT(var, 0.0);  // 5th percentile should be negative → VaR positive
}

// ─── TradeLog tests ───────────────────────────────────────────────────────────
void test_trade_log_pnl() {
    TradeLog log;
    Trade t1; t1.pnl = 1000; t1.grossPnl = 1100; t1.transactionCost = 100;
              t1.entryBar = 0; t1.exitBar = 5; t1.stoppedOut = false;
    Trade t2; t2.pnl = -500; t2.grossPnl = -400; t2.transactionCost = 100;
              t2.entryBar = 6; t2.exitBar = 10; t2.stoppedOut = false;
    log.addTrade(t1);
    log.addTrade(t2);
    log.finalise();

    EXPECT_NEAR(log.totalNetPnl(), 500.0, 1e-6);
    EXPECT_EQ(log.winningTrades(), 1);
    EXPECT_EQ(log.losingTrades(), 1);
    EXPECT_NEAR(log.winRate(), 0.5, 1e-6);
    EXPECT_NEAR(log.profitFactor(), 1000.0/500.0, 1e-6);
}

void test_trade_log_equity_curve() {
    TradeLog log;
    for (int i = 0; i < 10; ++i) {
        Trade t; t.pnl = 100; t.grossPnl = 110; t.transactionCost = 10;
                 t.entryBar = i*2; t.exitBar = i*2+1; t.stoppedOut = false;
        log.addTrade(t);
    }
    log.finalise(1'000'000.0);
    // Equity should be monotonically increasing
    const auto& eq = log.equityCurve();
    for (size_t i = 1; i < eq.size(); ++i)
        EXPECT_GT(eq[i], eq[i-1] - 1.0);
}

// ─── Integration test ─────────────────────────────────────────────────────────
void test_full_pipeline_cointegrated_pair() {
    // Build a strongly cointegrated pair and verify the system
    // finds it and generates some trades
    auto [A, B] = makeCointegratedPair(600, 1.5, 20.0, 3.0, 42);

    CointegrationEngine coint;
    auto pc = coint.engleGranger("A", "B", A, B);
    EXPECT_TRUE(pc.cointegrated);

    double delta = KalmanFilter::optimiseDelta(
        std::vector<double>(A.begin(), A.begin()+252),
        std::vector<double>(B.begin(), B.begin()+252), false);
    EXPECT_GT(delta, 0.0);

    KalmanConfig kcfg; kcfg.delta = delta;
    KalmanFilter kf(kcfg);
    kf.warmUp(std::vector<double>(A.begin(), A.begin()+252),
              std::vector<double>(B.begin(), B.begin()+252));

    auto spreadSeries = kf.spreadSeries(A, B);
    EXPECT_EQ((int)spreadSeries.size(), 600);

    SignalConfig scfg; scfg.zScoreWindow = 20;
    SignalGenerator sg(scfg);
    auto signals = sg.generateSignals(spreadSeries);

    int entryCount = 0;
    for (auto& s : signals) if (s.entrySignal) ++entryCount;
    EXPECT_GT(entryCount, 0);  // should have at least one trade entry
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║     STATISTICAL ARBITRAGE - UNIT TEST SUITE          ║\n";
    std::cout << "║     28 tests across all components                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    std::cout << "── Stats ──────────────────────────────────────────────\n";
    runTest("T01: Stats::mean and stddev",              test_stats_mean_std);
    runTest("T02: Stats::OLS (y=2x+1)",                test_stats_ols);
    runTest("T03: Stats::sharpeRatio",                 test_stats_sharpe);
    runTest("T04: Stats::maxDrawdown",                 test_stats_max_drawdown);
    runTest("T05: Hurst < 0.5 for mean-reverting OU",  test_hurst_mean_reverting);
    runTest("T06: Hurst ≈ 0.5 for random walk",        test_hurst_random_walk);

    std::cout << "\n── CointegrationEngine ─────────────────────────────────\n";
    runTest("T07: ADF rejects unit root (AR1, coef=0.4)", test_adf_rejects_stationary);
    runTest("T08: ADF cannot reject pure random walk",    test_adf_cannot_reject_rw);
    runTest("T09: EG detects cointegrated pair",          test_engle_granger_detects_coint);
    runTest("T10: EG runs clean on unrelated RWs",        test_engle_granger_rejects_unrelated);
    runTest("T11: Half-life estimation (OU, HL≈20)",      test_half_life_estimation);

    std::cout << "\n── KalmanFilter ─────────────────────────────────────────\n";
    runTest("T12: Kalman converges to true β=1.5",        test_kalman_tracks_beta);
    runTest("T13: Kalman spread near zero (A=2B)",        test_kalman_spread_near_zero);
    runTest("T14: Kalman P matrix stays positive",        test_kalman_positive_covariance);
    runTest("T15: Good delta gives higher log-likelihood", test_kalman_log_likelihood_increases_with_better_delta);

    std::cout << "\n── SignalGenerator ──────────────────────────────────────\n";
    runTest("T16: Entry signal fires at |z|>2",           test_signal_entry_at_2sigma);
    runTest("T17: Exit signal fires when |z|→0",          test_signal_exit_at_zero);
    runTest("T18: Stop signal fires at |z|>3.5",          test_signal_stop_at_3p5sigma);

    std::cout << "\n── PositionSizer ────────────────────────────────────────\n";
    runTest("T19: Position size is positive",              test_position_size_positive);
    runTest("T20: Position capped at maxAllocation",       test_position_size_capped);
    runTest("T21: Higher |z| → larger or equal position",  test_position_z_scaling);

    std::cout << "\n── RiskManager ──────────────────────────────────────────\n";
    runTest("T22: Blocks entry on portfolio drawdown",     test_risk_blocks_on_drawdown);
    runTest("T23: Blocks entry on max concentration",      test_risk_blocks_on_concentration);
    runTest("T24: Allows normal entry",                    test_risk_allows_normal);
    runTest("T25: VaR calculation positive",               test_var_calculation);
    runTest("T26: Trade log P&L and win rate",             test_trade_log_pnl);
    runTest("T27: Equity curve monotone on positive PnL",  test_trade_log_equity_curve);
    runTest("T28: Full pipeline on cointegrated pair",     test_full_pipeline_cointegrated_pair);

    std::cout << "\n══════════════════════════════════════════════════\n";
    std::cout << "  Results : " << passed_ << " passed, " << failed_ << " failed\n";
    if (failed_ == 0) std::cout << "  ✓  All tests passed!\n";
    else              std::cout << "  ✗  Some tests FAILED.\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    return failed_ == 0 ? 0 : 1;
}