#include "KalmanFilter.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>

// ─── init ─────────────────────────────────────────────────────────────────────
void KalmanFilter::init(const KalmanConfig& cfg) {
    config_ = cfg;
    Qval_   = cfg.delta;

    state_.beta          = cfg.beta0;
    state_.alpha         = cfg.alpha0;
    state_.P00           = cfg.P0;
    state_.P01           = 0.0;
    state_.P10           = 0.0;
    state_.P11           = cfg.P0;
    state_.Ve            = 1.0;   // will be estimated online
    state_.logLikelihood = 0.0;
    state_.barsProcessed = 0;

    betaHistory_.clear();
    alphaHistory_.clear();
}

// ─── Predict step ─────────────────────────────────────────────────────────────
// State transition: x_t = x_{t-1}   (random walk: no deterministic drift)
// Covariance update: P_t|t-1 = P_{t-1|t-1} + Q
// Q = delta * I  (scalar times identity for 2D state)
void KalmanFilter::predict() {
    // State stays the same (random walk prior)
    // Covariance grows by Q each step
    state_.P00 += Qval_;
    state_.P11 += Qval_;
    // P01, P10 unchanged (off-diagonal process noise assumed zero)
}

// ─── Update step ─────────────────────────────────────────────────────────────
// Observation model: priceA = [priceB, 1] * [β, α]' + ε
// H = [priceB, 1]   (observation matrix, row vector)
// Innovation: y_tilde = priceA - H * x_{t|t-1}
// S = H * P * H' + R   (innovation covariance, scalar)
// Kalman gain: K = P * H' / S
// x update: x = x + K * y_tilde
// P update: P = (I - K*H) * P
double KalmanFilter::updateStep(double priceA, double priceB) {
    // H = [priceB, 1]
    double h0 = priceB;
    double h1 = 1.0;

    // Predicted observation
    double pred = state_.beta * h0 + state_.alpha * h1;

    // Innovation
    double innov = priceA - pred;

    // Innovation covariance S = H * P * H' + Ve
    // = h0^2*P00 + h0*h1*(P01+P10) + h1^2*P11 + Ve
    double S = h0*h0*state_.P00 + h0*h1*(state_.P01 + state_.P10)
             + h1*h1*state_.P11 + state_.Ve;

    if (S < config_.minVariance) S = config_.minVariance;

    // Update observation noise estimate (running average)
    // Ve ≈ rolling mean of innovation^2 - H*P*H'
    state_.Ve = std::max(config_.minVariance,
                         0.95 * state_.Ve + 0.05 * innov * innov);

    // Kalman gain K = P * H' / S   (2×1 vector)
    double K0 = (state_.P00 * h0 + state_.P01 * h1) / S;
    double K1 = (state_.P10 * h0 + state_.P11 * h1) / S;

    // State update
    state_.beta  += K0 * innov;
    state_.alpha += K1 * innov;

    // Covariance update: P = (I - K*H) * P
    // New P = P - K * H * P
    double KH00 = K0 * h0;  double KH01 = K0 * h1;
    double KH10 = K1 * h0;  double KH11 = K1 * h1;

    double newP00 = state_.P00 - (KH00*state_.P00 + KH01*state_.P10);
    double newP01 = state_.P01 - (KH00*state_.P01 + KH01*state_.P11);
    double newP10 = state_.P10 - (KH10*state_.P00 + KH11*state_.P10);
    double newP11 = state_.P11 - (KH10*state_.P01 + KH11*state_.P11);

    // Symmetrise (numerical stability)
    state_.P00 = std::max(config_.minVariance, newP00);
    state_.P11 = std::max(config_.minVariance, newP11);
    state_.P01 = (newP01 + newP10) / 2.0;
    state_.P10 = state_.P01;

    // Log-likelihood contribution: -0.5 * (log(2π*S) + innov²/S)
    state_.logLikelihood += -0.5 * (std::log(2.0*M_PI*S) + innov*innov/S);
    ++state_.barsProcessed;

    return innov;
}

// ─── Public update ────────────────────────────────────────────────────────────
double KalmanFilter::update(double priceA, double priceB) {
    predict();
    double innov = updateStep(priceA, priceB);
    betaHistory_.push_back(state_.beta);
    alphaHistory_.push_back(state_.alpha);
    return innov;
}

// ─── Compute spread ───────────────────────────────────────────────────────────
double KalmanFilter::computeSpread(double priceA, double priceB) const {
    return priceA - state_.beta * priceB - state_.alpha;
}

// ─── Warm up ──────────────────────────────────────────────────────────────────
void KalmanFilter::warmUp(const std::vector<double>& priceA,
                           const std::vector<double>& priceB) {
    size_t n = std::min(priceA.size(), priceB.size());
    for (size_t i = 0; i < n; ++i) update(priceA[i], priceB[i]);
    std::cout << "[INFO] KalmanFilter warmed up on " << n << " bars."
              << " Final β=" << std::fixed << std::setprecision(4) << state_.beta
              << " α=" << state_.alpha << "\n";
}

// ─── Spread series ────────────────────────────────────────────────────────────
std::vector<double> KalmanFilter::spreadSeries(const std::vector<double>& priceA,
                                                 const std::vector<double>& priceB) {
    reset();
    size_t n = std::min(priceA.size(), priceB.size());
    std::vector<double> spreads(n);
    for (size_t i = 0; i < n; ++i) {
        update(priceA[i], priceB[i]);
        spreads[i] = computeSpread(priceA[i], priceB[i]);
    }
    return spreads;
}

// ─── Optimise delta (process noise) ──────────────────────────────────────────
// Grid search over delta values, maximise total log-likelihood
double KalmanFilter::optimiseDelta(const std::vector<double>& priceA,
                                    const std::vector<double>& priceB,
                                    bool verbose) {
    // Search grid: logarithmically spaced deltas
    std::vector<double> deltas = {1e-7, 5e-7, 1e-6, 5e-6,
                                   1e-5, 5e-5, 1e-4, 5e-4,
                                   1e-3, 5e-3, 1e-2};
    double bestDelta = 1e-4;
    double bestLL    = -INF_D;

    if (verbose) std::cout << "[Kalman delta search]\n";

    for (double d : deltas) {
        KalmanConfig cfg;
        cfg.delta = d;
        KalmanFilter kf(cfg);
        size_t n = std::min(priceA.size(), priceB.size());
        for (size_t i = 0; i < n; ++i) kf.update(priceA[i], priceB[i]);

        double ll = kf.getState().logLikelihood;
        if (verbose)
            std::cout << "  delta=" << std::scientific << d
                      << "  logL=" << std::fixed << std::setprecision(2) << ll << "\n";
        if (ll > bestLL) { bestLL = ll; bestDelta = d; }
    }

    if (verbose)
        std::cout << "  Best delta=" << std::scientific << bestDelta
                  << "  logL=" << bestLL << "\n\n";
    return bestDelta;
}