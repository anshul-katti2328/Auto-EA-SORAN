package network;

import orchestration.OrchestrationShim;
import simu5g.nodes.gnb.GNodeB;   // adjust path to your Simu5G module
import simu5g.nodes.ue.UE;

network EA_SORAN
{
    parameters:
        int numGnb = default(2);
        int numUe = default(4);

    submodules:
        orchShim: OrchestrationShim {
            parameters:
                zmqPort = 5558;
                pollInterval = 0.01;
        }

        gnb[numGnb]: GNodeB {
            @display("p=100,100+100*i");
        }

        ue[numUe]: UE {
            @display("p=300,100+50*i");
        }
}
