#include "ControllableGnbPhy.h"

#include <algorithm>

using namespace omnetpp;

Define_Module(ControllableGnbPhy);

void ControllableGnbPhy::initialize(
    int stage
) {
    simu5g::LtePhyEnbD2D::initialize(stage);

    /*
     * Simu5G selects macro or micro TX power during
     * initialization stage 1.
     */
    if (stage == 1) {
        configuredTxPowerDbm = txPower_;
        radioSleeping = false;
    }
}

void ControllableGnbPhy::applyOperationalState()
{
    /*
     * Simu5G places txPower_ into every transmitted radio
     * frame. A very low value therefore models an unavailable
     * sleeping radio without altering the protocol stack.
     */
    txPower_ =
        radioSleeping
            ? -200.0
            : configuredTxPowerDbm;

    // Keep the inherited cached reference powers consistent.
    eNodeBtxPower_ = configuredTxPowerDbm;
    microTxPower_ = configuredTxPowerDbm;
    EV_INFO
        << "ControllableGnbPhy: configured TX="
        << configuredTxPowerDbm
        << " dBm, sleeping="
        << (radioSleeping ? "true" : "false")
        << ", effective TX="
        << txPower_
        << " dBm\n";
}

void ControllableGnbPhy::setConfiguredTxPowerDbm(
    double txPowerDbm
) {
    configuredTxPowerDbm =
        std::clamp(
            txPowerDbm,
            5.0,
            43.0
        );

    applyOperationalState();
}

void ControllableGnbPhy::setRadioSleeping(
    bool sleeping
) {
    radioSleeping = sleeping;

    applyOperationalState();
}

double ControllableGnbPhy::
getConfiguredTxPowerDbm() const
{
    return configuredTxPowerDbm;
}

bool ControllableGnbPhy::
isRadioSleeping() const
{
    return radioSleeping;
}