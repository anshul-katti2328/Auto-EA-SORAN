package autoeasoran.zmqonly;

import simu5g.stack.phy.LtePhyEnbD2D;

// Runtime-controllable Simu5G gNB PHY.
//
// It retains the complete Simu5G 1.4.1 PHY implementation
// while exposing TX-power and radio-sleep control to the
// Auto EA-SORAN orchestration shim.
simple ControllableGnbPhy extends LtePhyEnbD2D
{
    parameters:
        @class(ControllableGnbPhy);
}