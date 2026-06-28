#ifndef __SHIM_REGISTRY_H_
#define __SHIM_REGISTRY_H_

#include <array>
#include <map>
#include <mutex>
#include <string>

// Process-local shared configuration store used by the
// OrchestrationShim and TwinMetricsAgent.
class ShimRegistry {
  public:
    struct BsConfig {
        double txPowerDbm = 42.0;
        std::string esMode = "active";
        double cioBiasDb = 0.0;

        // A value <= 0 means that TwinMetricsAgent should use
        // the base station's zone-profile default.
        double bandwidthMhz = 0.0;

        // Number of currently active component carriers.
        int carrierCount = 1;

        // Cell sleep, carrier off, MIMO reduction, Micro-DTX,
        // BWP adaptation, RF-chain sleep.
        std::array<int, 6> esModes = {
            0, 0, 0, 0, 0, 0
        };
    };

    static ShimRegistry& instance() {
        static ShimRegistry registry;
        return registry;
    }

    void setConfig(
        const std::string& bsId,
        const BsConfig& config
    ) {
        std::lock_guard<std::mutex> guard(mutex_);
        configurations_[bsId] = config;
    }

    bool getConfig(
        const std::string& bsId,
        BsConfig& output
    ) {
        std::lock_guard<std::mutex> guard(mutex_);

        auto iterator = configurations_.find(bsId);

        if (iterator == configurations_.end()) {
            return false;
        }

        output = iterator->second;
        return true;
    }

    std::map<std::string, BsConfig> snapshot() {
        std::lock_guard<std::mutex> guard(mutex_);
        return configurations_;
    }

  private:
    ShimRegistry() = default;

    std::mutex mutex_;
    std::map<std::string, BsConfig> configurations_;
};

#endif
