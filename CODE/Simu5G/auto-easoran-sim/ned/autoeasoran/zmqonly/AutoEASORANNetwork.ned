package autoeasoran.zmqonly;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;

import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.NrUe;
import simu5g.nodes.gNodeB;
import simu5g.nodes.Upf;
import simu5g.world.radio.LteChannelControl;

// ============================================================
// Auto EA-SORAN genuine Simu5G NR standalone topology.
//
// Default deployment:
//   30 real Simu5G gNodeBs
//   3 real NR UEs per gNodeB
//   90 total NR UEs
//
// The gNodeB and UE locations, cell associations and traffic
// applications are configured in omnetpp.ini.
// ============================================================
network AutoEASORANNetwork
{
    parameters:
        int numGnbs = default(30);
        int numUesPerGnb = default(3);
        int numUes = default(numGnbs * numUesPerGnb);

        @display("bgb=1400,950;bgi=background/terrain");

    submodules:
        channelControl: LteChannelControl {
            @display("p=55,40;is=s");
        }

        routingRecorder: RoutingTableRecorder {
            @display("p=55,85;is=s");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=55,130");
        }

        binder: Binder {
            @display("p=55,175;is=s");
        }

        carrierAggregation: CarrierAggregation {
            @display("p=55,220;is=s");
        }

        server: StandardHost {
            @display("p=180,70;i=device/server");
        }

        router: Router {
            @display("p=330,70;i=device/smallrouter");
        }

        upf: Upf {
            @display("p=480,70");
        }

        intermediateUpf: Upf {
            @display("p=630,70");
        }

        // Genuine Simu5G NR base stations.
        gnb[numGnbs]: gNodeB {
            @display("p=650,430;i=device/antennatower;is=l");
        }

        // Genuine Simu5G NR user equipment.
        ue[numUes]: NrUe {
            @display("p=650,580;i=device/cellphone;is=s");
        }

        shim: OrchestrationShim {
            @display("p=1120,850;i=block/app;t=Orchestration Shim");
        }

        metrics: TwinMetricsAgent {
            parameters:
                numGnbs = parent.numGnbs;

            @display("p=850,850;i=block/process;t=Twin Metrics");
        }

    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;

        router.pppg++ <--> Eth10G <--> upf.filterGate;

        upf.pppg++ <--> Eth10G <-->
            intermediateUpf.pppg++;

        for i = 0..numGnbs - 1 {
            intermediateUpf.pppg++ <--> Eth10G <-->
                gnb[i].ppp;
        }
}
