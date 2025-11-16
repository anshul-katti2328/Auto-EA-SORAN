package orchestration;

network MyNetwork
{
    submodules:
        shim: OrchestrationShim {
            parameters:
                zmqPort = 5558;  // must match orchestrationagent.py shim_socket.connect()
        }

        agent: OrchestrationAgent {
            parameters:
                zmqPort = 5555;  // must match optimization agent connect()
                metricsFile = "metrics.csv";
        }

        // Example: your gNodeB module
        gnb: <your_gNodeB_module> {
            parameters:
                // normal gNodeB params...
        }
}
