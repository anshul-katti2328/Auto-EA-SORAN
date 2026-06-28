#ifndef __TWIN_METRICS_AGENT_H_
#define __TWIN_METRICS_AGENT_H_

#include <omnetpp.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Zone metadata and energy-model parameters.
struct ZoneProfile {
    const char* zoneName;
    const char* zoneType;
    const char* bsType;
    const char* ruType;
    const char* mode;

    double frequencyMhz;
    int bandwidthMhz;
    int numAntennas;
    double baseTxPowerDbm;

    double pStaticW;
    double pMaxW;
    double deltaPower;

    double peakHour1;
    double peakHour2;
    double sigma1;
    double sigma2;
    double nightFloor;

    double basePrbPct;
    double peakPrbPct;

    int baseUes;
    int peakUes;

    double displayX;
    double displayY;
};

class TwinMetricsAgent :
    public omnetpp::cSimpleModule,
    public omnetpp::cListener
{
  private:
    enum class MetricKind {
        CELL_ALLOCATED_DL_BLOCKS,
        UE_GENERATED_DL_BYTES,
        UE_RECEIVED_DL_BYTES,
        UE_FRAME_DELAY,
        UE_SINR_DL
    };

    struct SignalBinding {
        MetricKind kind;
        int index;
        omnetpp::cComponent* source;
    };

    struct CellRuntimeMetrics {
        double allocatedDlBlocksSum = 0.0;
        std::uint64_t allocatedDlBlocksSamples = 0;
    };

    struct UeRuntimeMetrics {
        double generatedDlBytes = 0.0;
        double receivedDlBytes = 0.0;

        double delaySecondsSum = 0.0;
        std::uint64_t delaySamples = 0;

        double sinrDbSum = 0.0;
        std::uint64_t sinrSamples = 0;
    };

    using SignalKey =
        std::pair<int, omnetpp::simsignal_t>;

    std::string pubBind;
    std::string twinId;

    double publishInterval = 1.0;

    // 60 simulation seconds represent one logical day.
    double logicalTimeScale = 1440.0;

    int numGnbs = 30;
    int configuredPrbs = 51;

    bool echoToLog = false;

    omnetpp::cMessage* tick = nullptr;

    void* zmqCtx = nullptr;
    void* zmqPub = nullptr;

    std::vector<ZoneProfile> zoneProfiles;

    std::vector<CellRuntimeMetrics> cellRuntimeMetrics;
    std::vector<UeRuntimeMetrics> ueRuntimeMetrics;
    std::vector<omnetpp::cModule*> ueModules;

    std::map<SignalKey, SignalBinding> signalBindings;

    void initializeZoneProfiles();

    void subscribeToSimu5gSignals();

    void subscribeMetric(
        omnetpp::cComponent* source,
        const char* signalName,
        MetricKind kind,
        int index
    );

    void updateMetric(
        omnetpp::cComponent* source,
        omnetpp::simsignal_t signalId,
        double value
    );

    int servingCellIndex(int ueIndex) const;

    void resetIntervalMetrics();

  protected:
    virtual void initialize() override;

    virtual void handleMessage(
        omnetpp::cMessage* message
    ) override;

    virtual void finish() override;

    virtual void receiveSignal(
        omnetpp::cComponent* source,
        omnetpp::simsignal_t signalId,
        double value,
        omnetpp::cObject* details
    ) override;

    virtual void receiveSignal(
        omnetpp::cComponent* source,
        omnetpp::simsignal_t signalId,
        omnetpp::intval_t value,
        omnetpp::cObject* details
    ) override;

    virtual void receiveSignal(
        omnetpp::cComponent* source,
        omnetpp::simsignal_t signalId,
        omnetpp::uintval_t value,
        omnetpp::cObject* details
    ) override;

    virtual void receiveSignal(
        omnetpp::cComponent* source,
        omnetpp::simsignal_t signalId,
        const omnetpp::SimTime& value,
        omnetpp::cObject* details
    ) override;

    void ensureSockets();
    void publishTick();

    std::string snapshotJson();

    double trafficMultiplier(
        double logicalTimeSeconds,
        const ZoneProfile& zone
    );

    double calculateEnergy(
        const ZoneProfile& zone,
        double prbPct,
        double txPowerDbm,
        const std::string& esMode,
        int activeUes,
        double effectiveBandwidthMhz,
        int carrierCount,
        const std::array<int, 6>& esModes
    );

    double calculateLatency(
        double prbPct,
        double txPowerDbm,
        const std::string& esMode
    );

    double calculatePacketLoss(
        double prbPct,
        double sinrDb,
        double txPowerDbm
    );

    double calculateSpectralEfficiency(
        double sinrDb,
        int bandwidthMhz
    );

    int calculateAvailablePrbs(
        double bandwidthMhz,
        int carrierCount
    );

    std::string trafficLevelLabel(
        double multiplier
    );
};

#endif