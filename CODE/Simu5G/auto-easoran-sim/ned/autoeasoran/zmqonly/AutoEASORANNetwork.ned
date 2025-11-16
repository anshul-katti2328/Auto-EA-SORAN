package autoeasoran.zmqonly;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.StandardHost;

// Explicit gNB submodules (no vector), so Qtenv always shows seven icons.
network AutoEASORANNetwork
{
    @display("bgb=900,600");
    submodules:
        configurator: Ipv4NetworkConfigurator { @display("p=80,80"); }

        gnb0: StandardHost { @display("p=420,260;i=device/antennatower;t=CBD-1"); }
        gnb1: StandardHost { @display("p=320,180;i=device/antennatower;t=CBD-2"); }
        gnb2: StandardHost { @display("p=520,180;i=device/antennatower;t=CBD-3"); }
        gnb3: StandardHost { @display("p=600,280;i=device/antennatower;t=CBD-4"); }
        gnb4: StandardHost { @display("p=520,380;i=device/antennatower;t=CBD-5"); }
        gnb5: StandardHost { @display("p=320,380;i=device/antennatower;t=CBD-6"); }
        gnb6: StandardHost { @display("p=240,280;i=device/antennatower;t=CBD-7"); }

        shim: OrchestrationShim { @display("p=620,460;i=block/app;t=shim"); }
        metrics: TwinMetricsAgent {
            parameters:
                numGnbs = 7;   // tells the publisher how many BS entries to emit
            @display("p=360,460;i=block/process;t=metrics");
        }
}