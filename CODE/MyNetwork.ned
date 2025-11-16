package orchestration;

network MyNetwork
{
    submodules:
        shim: OrchestrationShim {
            parameters:
                zmqPort = 6000;  // ✅ FIXED - matches OrchestrationAgent.py
        }
        agent: OrchestrationAgent {
            parameters:
                zmqPort = 5555;  // ✅ CORRECT
                metricsFile = "metrics.csv";
        }
        gnb: <your_gNodeB_module> {
            parameters:
                // normal gNodeB params...
        }
}