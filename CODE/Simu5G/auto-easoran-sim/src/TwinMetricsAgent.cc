#include "TwinMetricsAgent.h"
#include "ShimRegistry.h"

#include <zmq.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <random>

using namespace omnetpp;

Define_Module(TwinMetricsAgent);

// ============================================================
// ISO8601 UTC timestamp
// ============================================================
static std::string nowUtcIso() {
    auto t  = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(t);
    std::tm gmt{};
#ifdef _WIN32
    gmtime_s(&gmt, &tt);
#else
    gmtime_r(&tt, &gmt);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);
    return buf;
}

// ============================================================
// Bangalore Tier-1 City Zone Profiles (30 gNBs)
// ============================================================
void TwinMetricsAgent::initializeZoneProfiles() {
    zoneProfiles.clear();
    zoneProfiles.resize(30);

    // ---- CBD Macro (0-6): MG Road, Brigade Road, Commercial Street ----
    // Heavy business traffic, peaks at morning (9AM) and evening (6PM)
    for (int i = 0; i <= 6; ++i) {
        zoneProfiles[i] = {
            "MG Road CBD",  "cbd",  "macro",  "Type1",  "Mode2",
            3500.0,  20,  4,  43.0,       // freq, bw, ant, txPower
            130.0,   200.0,  4.7,         // pStatic, pMax, delta
            9.5,     18.0,               // peakHour1 (9:30AM), peakHour2 (6PM)
            2.0,     2.5,                // sigma1, sigma2
            0.12,                        // nightFloor (12% at 3AM)
            40.0,    92.0,               // basePrb, peakPrb
            80,      450,                // baseUes, peakUes
            0, 0                         // display coords (set below)
        };
    }
    // Slight variation per BS within CBD
    zoneProfiles[0].displayX = 520; zoneProfiles[0].displayY = 340;
    zoneProfiles[1].displayX = 480; zoneProfiles[1].displayY = 290;
    zoneProfiles[2].displayX = 560; zoneProfiles[2].displayY = 290;
    zoneProfiles[3].displayX = 600; zoneProfiles[3].displayY = 350;
    zoneProfiles[4].displayX = 560; zoneProfiles[4].displayY = 400;
    zoneProfiles[5].displayX = 480; zoneProfiles[5].displayY = 400;
    zoneProfiles[6].displayX = 440; zoneProfiles[6].displayY = 340;

    // ---- IT Park Macro (7-11): Electronic City, Whitefield, ORR ----
    // Strong daytime bias, tech workers, peak at 10AM-1PM
    for (int i = 7; i <= 11; ++i) {
        zoneProfiles[i] = {
            "Electronic City IT Park",  "it_park",  "macro",  "Type2",  "Mode2",
            3500.0,  20,  4,  43.0,
            140.0,   250.0,  4.7,
            11.0,    15.5,                // peakHour1 (11AM), peakHour2 (3:30PM)
            2.5,     2.0,
            0.08,                         // Very low at night (IT parks empty)
            30.0,    88.0,
            50,      380,
            0, 0
        };
    }
    zoneProfiles[7].displayX  = 850; zoneProfiles[7].displayY  = 180;
    zoneProfiles[8].displayX  = 900; zoneProfiles[8].displayY  = 230;
    zoneProfiles[9].displayX  = 950; zoneProfiles[9].displayY  = 180;
    zoneProfiles[10].displayX = 900; zoneProfiles[10].displayY = 130;
    zoneProfiles[11].displayX = 850; zoneProfiles[11].displayY = 280;
    zoneProfiles[11].zoneName = "Whitefield IT Hub";

    // ---- Residential Macro (12-16): Koramangala, Jayanagar, Indiranagar ----
    // Evening/night dominant, streaming + IPL watching
    for (int i = 12; i <= 16; ++i) {
        zoneProfiles[i] = {
            "Koramangala Residential",  "residential",  "macro",  "Type4",  "Mode2",
            3500.0,  20,  2,  40.0,
            120.0,   180.0,  4.7,
            20.5,    9.0,                  // peakHour1 (8:30PM), peakHour2 (9AM)
            2.5,     1.5,
            0.18,                          // Some night activity (home users)
            25.0,    85.0,
            60,      320,
            0, 0
        };
    }
    zoneProfiles[12].displayX = 350; zoneProfiles[12].displayY = 520;
    zoneProfiles[12].zoneName = "Koramangala";
    zoneProfiles[13].displayX = 400; zoneProfiles[13].displayY = 570;
    zoneProfiles[13].zoneName = "Jayanagar";
    zoneProfiles[14].displayX = 450; zoneProfiles[14].displayY = 520;
    zoneProfiles[14].zoneName = "Indiranagar";
    zoneProfiles[15].displayX = 350; zoneProfiles[15].displayY = 620;
    zoneProfiles[15].zoneName = "JP Nagar";
    zoneProfiles[16].displayX = 450; zoneProfiles[16].displayY = 620;
    zoneProfiles[16].zoneName = "HSR Layout";

    // ---- Industrial/Suburban Macro (17-20): Peenya, Bommasandra ----
    // Shift-based, peaks at midday, very low at night
    for (int i = 17; i <= 20; ++i) {
        zoneProfiles[i] = {
            "Peenya Industrial",  "industrial",  "macro",  "Type7",  "Mode2",
            3500.0,  20,  2,  40.0,
            110.0,   150.0,  4.7,
            11.5,    14.5,                 // peakHour1 (11:30AM), peakHour2 (2:30PM)
            3.0,     2.0,
            0.06,                          // Very quiet at night
            15.0,    70.0,
            30,      150,
            0, 0
        };
    }
    zoneProfiles[17].displayX = 180; zoneProfiles[17].displayY = 180;
    zoneProfiles[17].zoneName = "Peenya Industrial";
    zoneProfiles[18].displayX = 230; zoneProfiles[18].displayY = 230;
    zoneProfiles[18].zoneName = "Yeshwanthpur";
    zoneProfiles[19].displayX = 250; zoneProfiles[19].displayY = 680;
    zoneProfiles[19].zoneName = "Bommasandra";
    zoneProfiles[20].displayX = 180; zoneProfiles[20].displayY = 730;
    zoneProfiles[20].zoneName = "Jigani Suburban";

    // ---- CBD Small Cells (21-23): UB City, Forum Mall, Brigade Gateway ----
    // Lunch rush + evening shopping, dense indoor coverage
    for (int i = 21; i <= 23; ++i) {
        zoneProfiles[i] = {
            "UB City Mall Hotspot",  "cbd",  "small_cell",  "Type6",  "Mode2",
            3500.0,  10,  2,  30.0,
            40.0,    50.0,  2.6,
            13.0,    19.0,                 // peakHour1 (1PM lunch), peakHour2 (7PM shopping)
            1.5,     2.0,
            0.05,                          // Malls close at night
            20.0,    95.0,
            100,     600,
            0, 0
        };
    }
    zoneProfiles[21].displayX = 540; zoneProfiles[21].displayY = 310;
    zoneProfiles[21].zoneName = "UB City Mall";
    zoneProfiles[22].displayX = 500; zoneProfiles[22].displayY = 370;
    zoneProfiles[22].zoneName = "Forum Mall Koramangala";
    zoneProfiles[23].displayX = 580; zoneProfiles[23].displayY = 370;
    zoneProfiles[23].zoneName = "Brigade Gateway";

    // ---- Stadium/Entertainment Small Cells (24-26): Chinnaswamy, Kanteerava ----
    // SPECIAL: IPL match spike 7:30PM-10:30PM, otherwise low
    for (int i = 24; i <= 26; ++i) {
        zoneProfiles[i] = {
            "M Chinnaswamy Stadium",  "stadium",  "small_cell",  "Type5",  "Mode2",
            3500.0,  10,  2,  30.0,
            45.0,    65.0,  2.6,
            20.0,    13.0,                 // peakHour1 (8PM IPL), peakHour2 (1PM day match)
            1.0,     2.0,                  // Very sharp IPL peak (sigma=1h)
            0.04,                          // Dead at night (no match)
            8.0,     98.0,                 // Massive swing: 8% → 98%
            30,      2500,                 // IPL: up to 2500 UEs!
            0, 0
        };
    }
    zoneProfiles[24].displayX = 650; zoneProfiles[24].displayY = 420;
    zoneProfiles[24].zoneName = "Chinnaswamy Stadium";
    zoneProfiles[25].displayX = 700; zoneProfiles[25].displayY = 470;
    zoneProfiles[25].zoneName = "Kanteerava Stadium";
    zoneProfiles[26].displayX = 650; zoneProfiles[26].displayY = 520;
    zoneProfiles[26].zoneName = "Cubbon Park Venue";

    // ---- Transit Hub Small Cells (27-29): Majestic, KR Puram, Kempegowda ----
    // Sharp bimodal: morning rush 7:30-9:30AM, evening rush 5:30-7:30PM
    for (int i = 27; i <= 29; ++i) {
        zoneProfiles[i] = {
            "Majestic Bus Station",  "transit",  "small_cell",  "Type3",  "Mode2",
            3500.0,  10,  4,  33.0,
            55.0,    75.0,  2.6,
            8.0,     18.0,                 // peakHour1 (8AM rush), peakHour2 (6PM rush)
            1.0,     1.0,                  // Very sharp commuter peaks
            0.10,
            15.0,    95.0,
            50,      800,
            0, 0
        };
    }
    zoneProfiles[27].displayX = 300; zoneProfiles[27].displayY = 340;
    zoneProfiles[27].zoneName = "Majestic Bus Terminal";
    zoneProfiles[28].displayX = 1000; zoneProfiles[28].displayY = 350;
    zoneProfiles[28].zoneName = "KR Puram Railway";
    zoneProfiles[29].displayX = 150; zoneProfiles[29].displayY = 450;
    zoneProfiles[29].zoneName = "Kempegowda Airport Rd";
}

// ============================================================
// Traffic multiplier: 24-hour pattern per zone
// Returns value in [0.0, 1.0]
// ============================================================
double TwinMetricsAgent::trafficMultiplier(double simTimeSec, const ZoneProfile& zone) {
    double hourOfDay = fmod(simTimeSec / 3600.0, 24.0);

    // Primary peak (Gaussian)
    double dh1 = hourOfDay - zone.peakHour1;
    // Handle wrap-around for peaks near midnight
    if (dh1 > 12.0) dh1 -= 24.0;
    if (dh1 < -12.0) dh1 += 24.0;
    double primary = exp(-dh1 * dh1 / (2.0 * zone.sigma1 * zone.sigma1));

    // Secondary peak
    double dh2 = hourOfDay - zone.peakHour2;
    if (dh2 > 12.0) dh2 -= 24.0;
    if (dh2 < -12.0) dh2 += 24.0;
    double secondary = 0.65 * exp(-dh2 * dh2 / (2.0 * zone.sigma2 * zone.sigma2));

    // Night trough: minimum around 3AM
    double dNight = hourOfDay - 3.0;
    if (dNight > 12.0) dNight -= 24.0;
    if (dNight < -12.0) dNight += 24.0;
    double nightFactor = zone.nightFloor + (1.0 - zone.nightFloor) *
                         (1.0 - exp(-dNight * dNight / 6.0));

    double combined = std::max(primary, secondary) * nightFactor;
    return std::clamp(combined, 0.03, 1.0);
}

// ============================================================
// Energy consumption model (3GPP-aligned)
// E = P_static + delta * P_max * (prb/100) + P_antenna * antennas
//   - ES mode savings applied on top
// ============================================================
double TwinMetricsAgent::calculateEnergy(
    const ZoneProfile& zone,
    double prbPct,
    double txPowerDbm,
    const std::string& esMode,
    int activeUes,
    double effectiveBandwidthMhz,
    int carrierCount,
    const std::array<int, 6>& esModes
) {
    const double load = std::clamp(prbPct / 100.0, 0.0, 1.0);

    const double bandwidthScale =
        effectiveBandwidthMhz /
        std::max(1.0, static_cast<double>(zone.bandwidthMhz));

    carrierCount = std::clamp(carrierCount, 1, 4);

    const double carrierScale =
        1.0 + 0.30 * static_cast<double>(carrierCount - 1);

    const double txPowerScale = std::clamp(
        std::pow(
            10.0,
            (txPowerDbm - zone.baseTxPowerDbm) / 10.0
        ),
        0.10,
        1.0
    );

    const double pDynamic =
        zone.deltaPower *
        zone.pMaxW *
        load *
        bandwidthScale *
        carrierScale *
        txPowerScale;

    double pAntenna =
        zone.numAntennas *
        2.5 *
        std::sqrt(txPowerScale);

    const double pBaseband =
        0.15 *
        activeUes *
        bandwidthScale;

    const double pAdditionalCarriers =
        8.0 * static_cast<double>(carrierCount - 1);

    double pTotal =
        zone.pStaticW +
        pDynamic +
        pAntenna +
        pBaseband +
        pAdditionalCarriers;

    // Cooling and auxiliary equipment overhead.
    pTotal *= 1.12;

    /*
     * The named ES mode is authoritative. The flag array is
     * used as a fallback when no named mode was supplied.
     */
    std::string effectiveEsMode = esMode;

    if (effectiveEsMode.empty() ||
        effectiveEsMode == "active" ||
        effectiveEsMode == "normal") {

        if (esModes[0] == 1)
            effectiveEsMode = "sleep";
        else if (esModes[1] == 1)
            effectiveEsMode = "carrier_off";
        else if (esModes[2] == 1)
            effectiveEsMode = "mimo_reduce";
        else if (esModes[3] == 1)
            effectiveEsMode = "micro_dtx";
        else if (esModes[4] == 1)
            effectiveEsMode = "bwp_adaptation";
        else if (esModes[5] == 1)
            effectiveEsMode = "light_sleep";
    }

    if (effectiveEsMode == "sleep") {
        pTotal *= 0.35;
    }
    else if (effectiveEsMode == "light_sleep" ||
             effectiveEsMode == "rf_chain_sleep") {
        pTotal *= 0.60;
    }
    else if (effectiveEsMode == "carrier_off") {
        pTotal *= 0.75;
    }
    else if (effectiveEsMode == "mimo_reduce") {
        pTotal *= 0.90;
    }
    else if (effectiveEsMode == "micro_dtx") {
        const double dtxFactor =
            1.0 - 0.30 * (1.0 - load);

        pTotal *= dtxFactor;
    }
    else if (effectiveEsMode == "bwp_adaptation") {
        pTotal *= 0.85;
    }

    return std::max(10.0, pTotal);
}

// ============================================================
// Latency model
// ============================================================
double TwinMetricsAgent::calculateLatency(double prbPct, double txPowerDbm,
                                           const std::string& esMode) {
    double load = prbPct / 100.0;

    // Base: 5G NR slot duration ~1ms, processing ~2ms
    double baseLatency = 3.0;

    // Queuing delay: exponential with load
    double queuingDelay = 2.0 * load * load / (1.0 - std::min(load, 0.95));
    queuingDelay = std::min(queuingDelay, 50.0); // cap at 50ms

    // Power impact (low power → retransmissions)
    double powerPenalty = std::max(0.0, (30.0 - txPowerDbm) * 0.15);

    // ES mode penalties
    double esPenalty = 0.0;
    if (esMode == "sleep")       esPenalty = 15.0;  // Wake-up latency
    else if (esMode == "light_sleep") esPenalty = 5.0;
    else if (esMode == "micro_dtx")   esPenalty = 2.0;
    else if (esMode == "carrier_off") esPenalty = 1.0;
    else if (esMode == "mimo_reduce") esPenalty = 1.5;

    return std::max(1.0, baseLatency + queuingDelay + powerPenalty + esPenalty);
}

// ============================================================
// Packet loss model
// ============================================================
double TwinMetricsAgent::calculatePacketLoss(double prbPct, double sinrDb,
                                              double txPowerDbm) {
    double load = prbPct / 100.0;

    double baseLoss = 0.0001; // 5G baseline

    // SINR impact (threshold-based)
    double sinrPenalty = 0.0;
    if (sinrDb < 3.0)       sinrPenalty = (3.0 - sinrDb) * 0.015;
    else if (sinrDb < 10.0) sinrPenalty = (10.0 - sinrDb) * 0.002;

    // Congestion
    double congestionPenalty = 0.0;
    if (load > 0.85) congestionPenalty = (load - 0.85) * 0.08;
    else if (load > 0.6) congestionPenalty = (load - 0.6) * 0.005;

    return std::clamp(baseLoss + sinrPenalty + congestionPenalty, 0.0, 0.10);
}

// ============================================================
// Spectral efficiency (Shannon-based approximation)
// ============================================================
double TwinMetricsAgent::calculateSpectralEfficiency(double sinrDb, int bandwidthMhz) {
    double sinrLinear = pow(10.0, sinrDb / 10.0);
    double se = log2(1.0 + sinrLinear);
    // Cap at practical 5G NR maximum (~7.4 bps/Hz for 256QAM)
    return std::clamp(se * 0.75, 0.1, 7.4); // 75% scheduling efficiency
}

// ============================================================
// Available NR PRBs using the 30 kHz SCS FR1 mapping.
// Values follow the standard NR transmission-bandwidth table.
// ============================================================
int TwinMetricsAgent::calculateAvailablePrbs(
    double bandwidthMhz,
    int carrierCount
) {
    static const double supportedBandwidths[] = {
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
        50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    };

    static const int prbsPerCarrier[] = {
        11, 24, 38, 51, 65, 78, 106,
        133, 162, 189, 217, 245, 273
    };

    const double clampedBandwidth =
        std::clamp(bandwidthMhz, 5.0, 100.0);

    const int clampedCarrierCount =
        std::clamp(carrierCount, 1, 4);

    int selectedPrbs = prbsPerCarrier[0];

    for (int i = 0; i < 13; ++i) {
        if (clampedBandwidth >= supportedBandwidths[i]) {
            selectedPrbs = prbsPerCarrier[i];
        }
        else {
            break;
        }
    }

    return selectedPrbs * clampedCarrierCount;
}

// ============================================================
// Traffic level label
// ============================================================
std::string TwinMetricsAgent::trafficLevelLabel(double multiplier) {
    if (multiplier < 0.20) return "very_low";
    if (multiplier < 0.40) return "low";
    if (multiplier < 0.60) return "medium";
    if (multiplier < 0.80) return "high";
    return "peak";
}

// ============================================================
// Simu5G runtime-signal collection
// ============================================================
void TwinMetricsAgent::subscribeMetric(
    cComponent* source,
    const char* signalName,
    MetricKind kind,
    int index
) {
    if (source == nullptr) {
        return;
    }

    const simsignal_t signalId =
        cComponent::registerSignal(signalName);

    const SignalKey key = {
        source->getId(),
        signalId
    };

    if (signalBindings.find(key) != signalBindings.end()) {
        return;
    }

    source->subscribe(signalId, this);

    signalBindings.emplace(
        key,
        SignalBinding{
            kind,
            index,
            source
        }
    );
}

void TwinMetricsAgent::subscribeToSimu5gSignals()
{
    cModule* network = getParentModule();

    if (network == nullptr) {
        throw cRuntimeError(
            "TwinMetricsAgent has no parent network"
        );
    }

    cellRuntimeMetrics.assign(
        static_cast<std::size_t>(numGnbs),
        CellRuntimeMetrics{}
    );

    const int numUes =
        network->getSubmoduleVectorSize("ue");

    ueRuntimeMetrics.assign(
        static_cast<std::size_t>(numUes),
        UeRuntimeMetrics{}
    );

    ueModules.clear();
    ueModules.reserve(
        static_cast<std::size_t>(numUes)
    );

    // Collect actual allocated downlink PRBs from every gNB.
    for (int cellIndex = 0;
         cellIndex < numGnbs;
         ++cellIndex) {

        cModule* gnb =
            network->getSubmodule(
                "gnb",
                cellIndex
            );

        if (gnb == nullptr) {
            EV_WARN
                << "TwinMetricsAgent: gNB "
                << cellIndex
                << " not found\n";

            continue;
        }

        cModule* cellularNic =
            gnb->getSubmodule("cellularNic");

        cModule* mac =
            cellularNic != nullptr
                ? cellularNic->getSubmodule("mac")
                : nullptr;

        subscribeMetric(
            mac,
            "avgServedBlocksDl",
            MetricKind::CELL_ALLOCATED_DL_BLOCKS,
            cellIndex
        );
    }

    // Collect received bytes, delay and measured SINR per UE.
    for (int ueIndex = 0;
         ueIndex < numUes;
         ++ueIndex) {

        cModule* ue =
            network->getSubmodule(
                "ue",
                ueIndex
            );

        ueModules.push_back(ue);

        if (ue == nullptr) {
            EV_WARN
                << "TwinMetricsAgent: UE "
                << ueIndex
                << " not found\n";

            continue;
        }

        cModule* receiver =
            ue->getSubmodule(
                "app",
                0
            );

        subscribeMetric(
            receiver,
            "cbrReceivedBytes",
            MetricKind::UE_RECEIVED_DL_BYTES,
            ueIndex
        );

        subscribeMetric(
            receiver,
            "cbrFrameDelay",
            MetricKind::UE_FRAME_DELAY,
            ueIndex
        );

        cModule* cellularNic =
            ue->getSubmodule("cellularNic");

        cModule* channelModel =
            cellularNic != nullptr
                ? cellularNic->getSubmodule(
                    "nrChannelModel",
                    0
                )
                : nullptr;

        subscribeMetric(
            channelModel,
            "rcvdSinrDl",
            MetricKind::UE_SINR_DL,
            ueIndex
        );
    }

    // Collect offered traffic from all server applications.
    cModule* server =
        network->getSubmodule("server");

    if (server != nullptr && numUes > 0) {
        const int numServerApps =
            server->getSubmoduleVectorSize("app");

        for (int appIndex = 0;
             appIndex < numServerApps;
             ++appIndex) {

            cModule* sender =
                server->getSubmodule(
                    "app",
                    appIndex
                );

            const int destinationUe =
                appIndex % numUes;

            subscribeMetric(
                sender,
                "cbrGeneratedBytes",
                MetricKind::UE_GENERATED_DL_BYTES,
                destinationUe
            );
        }
    }

    EV_INFO
        << "TwinMetricsAgent: subscribed to "
        << signalBindings.size()
        << " real Simu5G signal streams\n";
}

void TwinMetricsAgent::updateMetric(
    cComponent* source,
    simsignal_t signalId,
    double value
) {
    if (source == nullptr ||
        !std::isfinite(value)) {
        return;
    }

    const SignalKey key = {
        source->getId(),
        signalId
    };

    const auto bindingIterator =
        signalBindings.find(key);

    if (bindingIterator == signalBindings.end()) {
        return;
    }

    const SignalBinding& binding =
        bindingIterator->second;

    switch (binding.kind) {
        case MetricKind::CELL_ALLOCATED_DL_BLOCKS: {
            if (binding.index < 0 ||
                binding.index >=
                    static_cast<int>(
                        cellRuntimeMetrics.size()
                    )) {
                return;
            }

            CellRuntimeMetrics& metrics =
                cellRuntimeMetrics[
                    static_cast<std::size_t>(
                        binding.index
                    )
                ];

            metrics.allocatedDlBlocksSum +=
                std::max(0.0, value);

            ++metrics.allocatedDlBlocksSamples;
            break;
        }

        case MetricKind::UE_GENERATED_DL_BYTES: {
            if (binding.index < 0 ||
                binding.index >=
                    static_cast<int>(
                        ueRuntimeMetrics.size()
                    )) {
                return;
            }

            ueRuntimeMetrics[
                static_cast<std::size_t>(
                    binding.index
                )
            ].generatedDlBytes +=
                std::max(0.0, value);

            break;
        }

        case MetricKind::UE_RECEIVED_DL_BYTES: {
            if (binding.index < 0 ||
                binding.index >=
                    static_cast<int>(
                        ueRuntimeMetrics.size()
                    )) {
                return;
            }

            ueRuntimeMetrics[
                static_cast<std::size_t>(
                    binding.index
                )
            ].receivedDlBytes +=
                std::max(0.0, value);

            break;
        }

        case MetricKind::UE_FRAME_DELAY: {
            if (binding.index < 0 ||
                binding.index >=
                    static_cast<int>(
                        ueRuntimeMetrics.size()
                    )) {
                return;
            }

            UeRuntimeMetrics& metrics =
                ueRuntimeMetrics[
                    static_cast<std::size_t>(
                        binding.index
                    )
                ];

            metrics.delaySecondsSum +=
                std::max(0.0, value);

            ++metrics.delaySamples;
            break;
        }

        case MetricKind::UE_SINR_DL: {
            if (binding.index < 0 ||
                binding.index >=
                    static_cast<int>(
                        ueRuntimeMetrics.size()
                    )) {
                return;
            }

            UeRuntimeMetrics& metrics =
                ueRuntimeMetrics[
                    static_cast<std::size_t>(
                        binding.index
                    )
                ];

            metrics.sinrDbSum += value;
            ++metrics.sinrSamples;
            break;
        }
    }
}

void TwinMetricsAgent::receiveSignal(
    cComponent* source,
    simsignal_t signalId,
    double value,
    cObject* details
) {
    (void)details;

    updateMetric(
        source,
        signalId,
        value
    );
}

void TwinMetricsAgent::receiveSignal(
    cComponent* source,
    simsignal_t signalId,
    intval_t value,
    cObject* details
) {
    (void)details;

    updateMetric(
        source,
        signalId,
        static_cast<double>(value)
    );
}

void TwinMetricsAgent::receiveSignal(
    cComponent* source,
    simsignal_t signalId,
    uintval_t value,
    cObject* details
) {
    (void)details;

    updateMetric(
        source,
        signalId,
        static_cast<double>(value)
    );
}

void TwinMetricsAgent::receiveSignal(
    cComponent* source,
    simsignal_t signalId,
    const SimTime& value,
    cObject* details
) {
    (void)details;

    updateMetric(
        source,
        signalId,
        value.dbl()
    );
}

int TwinMetricsAgent::servingCellIndex(
    int ueIndex
) const {
    if (ueIndex < 0 ||
        ueIndex >=
            static_cast<int>(ueModules.size())) {
        return -1;
    }

    cModule* ue =
        ueModules[
            static_cast<std::size_t>(ueIndex)
        ];

    if (ue == nullptr) {
        return -1;
    }

    int servingNodeId = 0;

    // Simu5G LtePhyUe updates nrMasterId after dynamic
    // cell association for an NrUe.
    if (ue->hasPar("nrMasterId")) {
        servingNodeId =
            ue->par("nrMasterId").intValue();
    }

    // LTE fallback.
    if (servingNodeId <= 0 &&
        ue->hasPar("masterId")) {
        servingNodeId =
            ue->par("masterId").intValue();
    }

    const int cellIndex = servingNodeId - 1;

    if (cellIndex < 0 ||
        cellIndex >= numGnbs) {
        return -1;
    }

    return cellIndex;
}

void TwinMetricsAgent::resetIntervalMetrics()
{
    for (CellRuntimeMetrics& metrics :
         cellRuntimeMetrics) {

        metrics.allocatedDlBlocksSum = 0.0;
        metrics.allocatedDlBlocksSamples = 0;
    }

    for (UeRuntimeMetrics& metrics :
         ueRuntimeMetrics) {

        metrics.generatedDlBytes = 0.0;
        metrics.receivedDlBytes = 0.0;

        metrics.delaySecondsSum = 0.0;
        metrics.delaySamples = 0;

        metrics.sinrDbSum = 0.0;
        metrics.sinrSamples = 0;
    }
}

// ============================================================
// Initialization
// ============================================================
void TwinMetricsAgent::initialize()
{
    pubBind =
        par("pubBind").stdstringValue();

    publishInterval =
        par("publishInterval").doubleValue();

    numGnbs =
        par("numGnbs").intValue();

    twinId =
        par("twinId").stdstringValue();

    echoToLog =
        par("echoToLog").boolValue();

    initializeZoneProfiles();

    // Read the actual number of configured NR resource blocks.
    cModule* network = getParentModule();

    if (network != nullptr) {
        cModule* carrierAggregation =
            network->getSubmodule(
                "carrierAggregation"
            );

        cModule* componentCarrier =
            carrierAggregation != nullptr
                ? carrierAggregation->getSubmodule(
                    "componentCarrier",
                    0
                )
                : nullptr;

        if (componentCarrier != nullptr &&
            componentCarrier->hasPar("numBands")) {

            configuredPrbs =
                componentCarrier
                    ->par("numBands")
                    .intValue();
        }
    }

    subscribeToSimu5gSignals();

    tick = new cMessage("tick");

    // Begin publishing after the configured warm-up period.
    scheduleAt(
        getSimulation()->getWarmupPeriod() +
            publishInterval,
        tick
    );
}

// ============================================================
// ZMQ socket setup
// ============================================================
void TwinMetricsAgent::ensureSockets() {
    if (!zmqCtx) zmqCtx = zmq_ctx_new();
    if (!zmqPub) {
        zmqPub = zmq_socket(zmqCtx, ZMQ_PUB);
        if (zmq_bind(zmqPub, pubBind.c_str()) != 0)
            throw cRuntimeError("TwinMetricsAgent: PUB bind failed on %s", pubBind.c_str());
        EV_INFO << "TwinMetricsAgent: Metrics PUB bound to " << pubBind << "\n";
    }
}

// ============================================================
// Main JSON snapshot — the heart of the simulation
// ============================================================
std::string TwinMetricsAgent::snapshotJson()
{
    const double physicalTimeSeconds =
        SIMTIME_DBL(simTime());

    const double logicalTimeSeconds =
        physicalTimeSeconds * logicalTimeScale;

    const double hourOfDay =
        std::fmod(
            logicalTimeSeconds / 3600.0,
            24.0
        );

    const std::string wallTime =
        nowUtcIso();

    const int numBs =
        std::min(
            numGnbs,
            static_cast<int>(
                zoneProfiles.size()
            )
        );

    std::vector<double> generatedBytes(
        static_cast<std::size_t>(numBs),
        0.0
    );

    std::vector<double> receivedBytes(
        static_cast<std::size_t>(numBs),
        0.0
    );

    std::vector<double> delaySecondsSum(
        static_cast<std::size_t>(numBs),
        0.0
    );

    std::vector<std::uint64_t> delaySamples(
        static_cast<std::size_t>(numBs),
        0
    );

    std::vector<double> sinrDbSum(
        static_cast<std::size_t>(numBs),
        0.0
    );

    std::vector<std::uint64_t> sinrSamples(
        static_cast<std::size_t>(numBs),
        0
    );

    std::vector<int> activeUes(
        static_cast<std::size_t>(numBs),
        0
    );

    int unassociatedUes = 0;

    // Aggregate real UE measurements according to their
    // serving Simu5G cell.
    for (int ueIndex = 0;
         ueIndex <
             static_cast<int>(
                 ueRuntimeMetrics.size()
             );
         ++ueIndex) {

        const int cellIndex =
            servingCellIndex(ueIndex);

        if (cellIndex < 0 ||
            cellIndex >= numBs) {

            ++unassociatedUes;
            continue;
        }

        const UeRuntimeMetrics& ueMetrics =
            ueRuntimeMetrics[
                static_cast<std::size_t>(
                    ueIndex
                )
            ];

        const std::size_t cellOffset =
            static_cast<std::size_t>(
                cellIndex
            );

        ++activeUes[cellOffset];

        generatedBytes[cellOffset] +=
            ueMetrics.generatedDlBytes;

        receivedBytes[cellOffset] +=
            ueMetrics.receivedDlBytes;

        delaySecondsSum[cellOffset] +=
            ueMetrics.delaySecondsSum;

        delaySamples[cellOffset] +=
            ueMetrics.delaySamples;

        sinrDbSum[cellOffset] +=
            ueMetrics.sinrDbSum;

        sinrSamples[cellOffset] +=
            ueMetrics.sinrSamples;
    }

    const auto configurations =
        ShimRegistry::instance().snapshot();

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3);

    stream << "{";
    stream << "\"topic\":\"all/tick\",";
    stream << "\"schema_version\":3,";
    stream << "\"twin_id\":\""
           << twinId << "\",";

    // Logical time is consumed by the Python agents.
    stream << "\"sim_time_s\":"
           << logicalTimeSeconds << ",";

    stream << "\"physical_sim_time_s\":"
           << physicalTimeSeconds << ",";

    stream << "\"logical_time_scale\":"
           << logicalTimeScale << ",";

    stream << "\"wall_time_utc\":\""
           << wallTime << "\",";

    stream << "\"hour_of_day\":"
           << hourOfDay << ",";

    stream << "\"unassociated_ues\":"
           << unassociatedUes << ",";

    stream << "\"transient\":false,";
    stream << "\"kpi_source\":\"simu5g_runtime_signals\",";
    stream << "\"energy_source\":\"analytical_load_model\",";

    stream << "\"cells\":[";

    bool firstCell = true;

    for (int cellIndex = 0;
         cellIndex < numBs;
         ++cellIndex) {

        const ZoneProfile& zone =
            zoneProfiles[
                static_cast<std::size_t>(
                    cellIndex
                )
            ];

        const std::size_t offset =
            static_cast<std::size_t>(
                cellIndex
            );

        const std::string bsId =
            std::to_string(
                1000 + cellIndex + 1
            );

        const CellRuntimeMetrics& radioMetrics =
            cellRuntimeMetrics[offset];

        const double averageAllocatedPrbs =
            radioMetrics.allocatedDlBlocksSamples > 0
                ? radioMetrics.allocatedDlBlocksSum /
                    static_cast<double>(
                        radioMetrics
                            .allocatedDlBlocksSamples
                    )
                : 0.0;

        const int availablePrbs =
            std::max(1, configuredPrbs);

        // Mean occupied PRBs can be fractional because this is
        // averaged across all TTIs in the reporting interval.
        const double usedPrbs =
            std::clamp(
                averageAllocatedPrbs,
                0.0,
                static_cast<double>(availablePrbs)
            );

        const double prbUtilizationPct =
            100.0 *
            usedPrbs /
            static_cast<double>(availablePrbs);

        const double normalizedLoad =
            prbUtilizationPct / 100.0;

        const double generatedDlMbps =
            generatedBytes[offset] *
            8.0 /
            publishInterval /
            1000000.0;

        const double dlMbps =
            receivedBytes[offset] *
            8.0 /
            publishInterval /
            1000000.0;

        const double ulMbps = 0.0;

        const double totalThroughputMbps =
            dlMbps + ulMbps;

        const double latencyMs =
            delaySamples[offset] > 0
                ? delaySecondsSum[offset] /
                    static_cast<double>(
                        delaySamples[offset]
                    ) *
                    1000.0
                : 0.0;

        const double sinrDb =
            sinrSamples[offset] > 0
                ? sinrDbSum[offset] /
                    static_cast<double>(
                        sinrSamples[offset]
                    )
                : 0.0;

        double packetLossPct = 0.0;

        if (generatedBytes[offset] > 0.0) {
            packetLossPct =
                100.0 *
                (
                    generatedBytes[offset] -
                    receivedBytes[offset]
                ) /
                generatedBytes[offset];

            packetLossPct =
                std::clamp(
                    packetLossPct,
                    0.0,
                    100.0
                );
        }

        const double spectralEfficiency =
            sinrSamples[offset] > 0
                ? calculateSpectralEfficiency(
                    sinrDb,
                    20
                )
                : 0.0;

        ShimRegistry::BsConfig configuration;

        const auto configurationIterator =
            configurations.find(bsId);

        if (configurationIterator !=
            configurations.end()) {

            configuration =
                configurationIterator->second;
        }
        else {
            configuration.txPowerDbm =
                cellIndex < 21
                    ? 43.0
                    : 30.0;

            configuration.esMode = "active";
            configuration.bandwidthMhz = 20.0;
            configuration.carrierCount = 1;
            configuration.esModes =
                {0, 0, 0, 0, 0, 0};
        }

        const double physicalBandwidthMhz =
            20.0;

        const int physicalCarrierCount = 1;

        const bool radioSleeping =
    configuration.esMode == "sleep" ||
    configuration.esModes[0] == 1;

const double physicalTxPowerDbm =
    radioSleeping
        ? -200.0
        : configuration.txPowerDbm;

        const double energyW =
            calculateEnergy(
                zone,
                prbUtilizationPct,
                configuration.txPowerDbm,
                configuration.esMode,
                activeUes[offset],
                physicalBandwidthMhz,
                physicalCarrierCount,
                configuration.esModes
            );

        // Physical energy consumed during the accelerated
        // simulator interval.
        const double intervalEnergyJ =
            energyW * publishInterval;

        // Logical real-world interval represented by this
        // accelerated simulator sample.
        const double representedIntervalSeconds =
            publishInterval * logicalTimeScale;

        // Energy represented in the compressed 24-hour study.
        const double representedEnergyJ =
            energyW * representedIntervalSeconds;

        const double energyEfficiencyMbitPerJ =
            energyW > 0.0
                ? totalThroughputMbps / energyW
                : 0.0;

        const long long dlBits =
            static_cast<long long>(
                std::llround(
                    receivedBytes[offset] *
                    8.0
                )
            );

        const long long ulBits = 0;

        // Normalise offered traffic against the configured
        // 3 Mbps-per-UE evening-event rate.
        const double peakOfferedMbps =
            std::max(
                3.0,
                3.0 *
                static_cast<double>(
                    activeUes[offset]
                )
            );

        const double trafficRatio =
            std::clamp(
                generatedDlMbps /
                    peakOfferedMbps,
                0.0,
                1.0
            );

        const std::string trafficLevel =
            trafficLevelLabel(
                trafficRatio
            );

        if (!firstCell) {
            stream << ",";
        }

        firstCell = false;

        stream << "{";

        stream << "\"bs_id\":\""
               << bsId << "\",";

        stream << "\"simu5g_cell_index\":"
               << cellIndex << ",";

        stream << "\"zone\":\""
               << zone.zoneType << "\",";

        stream << "\"zone_name\":\""
               << zone.zoneName << "\",";

        stream << "\"bs_type\":\""
               << zone.bsType << "\",";

        stream << "\"ru_type\":\""
               << zone.ruType << "\",";

        stream << "\"mode\":\""
               << zone.mode << "\",";

        stream << "\"frequency_mhz\":3500.0,";
        stream << "\"bandwidth_mhz\":20.0,";
        stream << "\"subcarrier_spacing_khz\":30,";
        stream << "\"carrier_count\":1,";

        stream << "\"requested_bandwidth_mhz\":"
               << configuration.bandwidthMhz << ",";

        stream << "\"requested_carrier_count\":"
               << configuration.carrierCount << ",";

        stream << "\"num_antennas\":"
               << zone.numAntennas << ",";

        stream << "\"available_prbs\":"
               << availablePrbs << ",";

        stream << "\"used_prbs\":"
               << usedPrbs << ",";

        stream << "\"prb_util_pct\":"
               << prbUtilizationPct << ",";

        stream << "\"load\":"
               << normalizedLoad << ",";

        stream << "\"generated_dl_mbps\":"
               << generatedDlMbps << ",";

        stream << "\"dl_mbps\":"
               << dlMbps << ",";

        stream << "\"ul_mbps\":"
               << ulMbps << ",";

        stream << "\"total_throughput_mbps\":"
               << totalThroughputMbps << ",";

        stream << "\"measurement_interval_s\":"
               << publishInterval << ",";

        stream << "\"dl_bits\":"
               << dlBits << ",";

        stream << "\"ul_bits\":"
               << ulBits << ",";

        stream << "\"active_ues\":"
               << activeUes[offset] << ",";

        stream << "\"avg_sinr_db\":"
               << sinrDb << ",";

        stream << "\"avg_latency_ms\":"
               << latencyMs << ",";

        stream << "\"packet_loss_pct\":"
               << packetLossPct << ",";

        stream << "\"spectral_efficiency_bps_hz\":"
               << spectralEfficiency << ",";

        stream << "\"tx_power_dbm\":"
               << configuration.txPowerDbm << ",";

        stream << "\"physical_tx_power_dbm\":"
               << physicalTxPowerDbm << ",";

        stream << "\"radio_sleeping\":"
               << (radioSleeping ? "true" : "false")
               << ",";

        stream << "\"es_mode\":\""
               << configuration.esMode << "\",";

        stream << "\"es_modes\":[";

        for (std::size_t modeIndex = 0;
             modeIndex <
                 configuration.esModes.size();
             ++modeIndex) {

            if (modeIndex > 0) {
                stream << ",";
            }

            stream <<
                configuration.esModes[modeIndex];
        }

        stream << "],";

        stream << "\"energy_consumption_w\":"
               << energyW << ",";

        stream << "\"interval_energy_j\":"
               << intervalEnergyJ << ",";

        stream << "\"represented_interval_s\":"
               << representedIntervalSeconds << ",";

        stream << "\"represented_energy_j\":"
               << representedEnergyJ << ",";

        stream <<
            "\"energy_efficiency_mbit_per_j\":"
            << energyEfficiencyMbitPerJ << ",";

        stream << "\"hour_of_day\":"
               << hourOfDay << ",";

        stream << "\"traffic_level\":\""
               << trafficLevel << "\",";

        stream << "\"traffic_multiplier\":"
               << trafficRatio << ",";

        stream << "\"kpi_source\":"
               << "\"simu5g_runtime_signals\"";

        stream << "}";
    }

    stream << "]";
    stream << "}";

    const std::string payload =
        stream.str();

    resetIntervalMetrics();

    return payload;
}

// ============================================================
// Publish & event loop
// ============================================================
void TwinMetricsAgent::publishTick() {
    std::string payload = snapshotJson();

    zmq_msg_t msg;
    zmq_msg_init_size(&msg, payload.size());
    memcpy(zmq_msg_data(&msg), payload.data(), payload.size());
    zmq_msg_send(&msg, zmqPub, 0);
    zmq_msg_close(&msg);

    if (echoToLog)
        EV_INFO << "TwinMetricsAgent: published tick for " << numGnbs
                << " cells at sim_time=" << simTime() << "\n";
}

void TwinMetricsAgent::handleMessage(cMessage* msg) {
    if (msg == tick) {
        try {
            ensureSockets();
            publishTick();
        } catch (std::exception& e) {
            EV_WARN << "TwinMetricsAgent: publishTick error: " << e.what() << "\n";
        }
        scheduleAt(simTime() + publishInterval, tick);
    }
}

void TwinMetricsAgent::finish()
{
    for (const auto& entry : signalBindings) {
        const simsignal_t signalId =
            entry.first.second;

        cComponent* source =
            entry.second.source;

        if (source != nullptr &&
            source->isSubscribed(
                signalId,
                this
            )) {

            source->unsubscribe(
                signalId,
                this
            );
        }
    }

    signalBindings.clear();
    ueModules.clear();

    if (tick != nullptr) {
        cancelAndDelete(tick);
        tick = nullptr;
    }

    if (zmqPub != nullptr) {
        zmq_close(zmqPub);
        zmqPub = nullptr;
    }

    if (zmqCtx != nullptr) {
        zmq_ctx_term(zmqCtx);
        zmqCtx = nullptr;
    }
}