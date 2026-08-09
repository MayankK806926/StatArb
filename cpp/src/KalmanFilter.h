#pragma once
#include "Utils.h"

// ─── KalmanState ──────────────────────────────────────────────────────────────
// State vector x = [β, α]  (hedge ratio, intercept)
// Observation model:  priceA = β * priceB + α + ε
// State transition:   x_{t} = x_{t-1} + w   (random walk prior on β)
//
// This is the Kalman Filter described in Avellaneda & Lee (2010)
// and widely used in industrial pairs trading implementations.
// ─────────────────────────────────────────────────────────────────────────────
struct KalmanState {
    double beta;        // current hedge ratio estimate
    double alpha;       // current intercept estimate
    double P00, P01, P10, P11;  // 2x2 error covariance matrix
    double Ve;          // observation noise variance (R)
    double logLikelihood;
    int    barsProcessed;
};

// ─── KalmanConfig ─────────────────────────────────────────────────────────────
struct KalmanConfig {
    // Process noise covariance matrix Q (how fast β and α can change)
    // Larger delta = β changes faster = more responsive but noisier
    double delta       = 1e-4;   // drives Q = delta * I
    // Initial error covariance (large = diffuse prior)
    double P0          = 1.0;
    // Initial state
    double beta0       = 1.0;
    double alpha0      = 0.0;
    // Minimum variance floor (prevents numerical issues)
    double minVariance = 1e-8;
};

// ─── KalmanFilter ─────────────────────────────────────────────────────────────
// Online 2-state Kalman filter for dynamic hedge ratio estimation.
//
// Usage:
//   KalmanFilter kf;
//   kf.init(config);
//   for each bar: kf.update(priceA, priceB);
//   double beta = kf.getState().beta;
//   double spread = priceA - beta*priceB - kf.getState().alpha;
// ─────────────────────────────────────────────────────────────────────────────
class KalmanFilter {
public:
    KalmanFilter() = default;
    explicit KalmanFilter(const KalmanConfig& cfg) { init(cfg); }

    void init(const KalmanConfig& cfg = KalmanConfig{});

    // Process one new observation (priceA, priceB)
    // Returns the innovation (observed - predicted spread)
    double update(double priceA, double priceB);

    // Batch initialisation on historical data
    // (warms up the filter before live trading)
    void warmUp(const std::vector<double>& priceA,
                const std::vector<double>& priceB);

    // Current state estimate
    const KalmanState& getState() const { return state_; }

    // Compute spread using current β and α
    double computeSpread(double priceA, double priceB) const;

    // Get full spread series for a historical window
    std::vector<double> spreadSeries(const std::vector<double>& priceA,
                                      const std::vector<double>& priceB);

    // Get β history (populated during warmUp or spreadSeries)
    const std::vector<double>& getBetaHistory() const { return betaHistory_; }

    // Optimise delta (Q parameter) by maximising log-likelihood
    // Searches over a grid of delta values, returns best delta
    static double optimiseDelta(const std::vector<double>& priceA,
                                 const std::vector<double>& priceB,
                                 bool verbose = false);

    void reset() { init(config_); }

private:
    KalmanConfig        config_;
    KalmanState         state_;
    std::vector<double> betaHistory_;
    std::vector<double> alphaHistory_;
    double              Qval_;   // process noise = delta * I

    // Predict step: propagate state forward (state transition is identity)
    void predict();

    // Update step: incorporate new observation
    double updateStep(double priceA, double priceB);
};