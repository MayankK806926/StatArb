cmake_minimum_required(VERSION 3.16)
project(StatArb VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release build with optimisation flags ────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# NOTE: intentionally no -march=native - it bakes in the build machine's
# CPU features, so a Release binary built on one machine can SIGILL when
# copied to and run on a different (older/different-vendor) CPU.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -Wpedantic -fsanitize=address,undefined")

# ── Source files ─────────────────────────────────────────────────────────────
set(SRCS
    src/DataFetcher.cpp
    src/CointegrationEngine.cpp
    src/KalmanFilter.cpp
    src/SignalGenerator.cpp
    src/PositionSizer.cpp
    src/RiskManager.cpp
    src/TradeLog.cpp
    src/WalkForwardEngine.cpp
    src/ReportGenerator.cpp
)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(statarb src/main.cpp ${SRCS})
target_include_directories(statarb PRIVATE src)
target_compile_options(statarb PRIVATE -Wall -Wextra)

# ── Unit tests ────────────────────────────────────────────────────────────────
add_executable(test_statarb tests/test_all.cpp ${SRCS})
target_include_directories(test_statarb PRIVATE src)
target_compile_options(test_statarb PRIVATE -Wall -Wextra)

# ── CTest integration ─────────────────────────────────────────────────────────
enable_testing()
add_test(NAME unit_tests COMMAND test_statarb)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS statarb DESTINATION bin)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "Targets    : statarb, test_statarb")