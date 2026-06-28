#ifndef __CONTROLLABLE_GNB_PHY_H_
#define __CONTROLLABLE_GNB_PHY_H_

#include "simu5g/stack/phy/LtePhyEnbD2D.h"

class ControllableGnbPhy :
    public simu5g::LtePhyEnbD2D
{
  private:
    double configuredTxPowerDbm = 43.0;
    bool radioSleeping = false;

    void applyOperationalState();

  protected:
    virtual void initialize(
        int stage
    ) override;

  public:
    void setConfiguredTxPowerDbm(
        double txPowerDbm
    );

    void setRadioSleeping(
        bool sleeping
    );

    double getConfiguredTxPowerDbm() const;

    bool isRadioSleeping() const;
};

#endif