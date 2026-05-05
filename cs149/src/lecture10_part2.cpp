// lecture10_part2.cpp
// Roofline Model & Arithmetic Intensity Analysis
// Models the performance ceiling of compute vs memory bandwidth
// Stanford CS149, Fall 2025 - Lecture 10: Hardware Specialization

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

// Represents a hardware platform's performance limits
struct HardwareSpec {
    std::string name;
    double peakTFLOPS;          // Peak compute (TFLOPS)
    double memoryBandwidthGBs;  // Peak memory bandwidth (GB/s)
    double onChipSRAM_MB;       // On-chip SRAM capacity (MB)
    double perfPerWatt;         // Relative efficiency (higher = better)
};

// Represents a computational kernel's characteristics
struct Kernel {
    std::string name;
    double flops;               // Total floating-point operations
    double bytesAccessed;       // Total bytes accessed from off-chip memory
    // Arithmetic intensity = FLOPs / bytes
    double arithmeticIntensity() const {
        return flops / bytesAccessed;
    }
};

// Roofline model: determines if a kernel is compute-bound or memory-bound
struct RooflineResult {
    double arithmeticIntensity;
    double achievableTFLOPS;
    double ridgePoint;          // AI value where compute == memory limit
    bool isComputeBound;
    double utilizationPercent;
};

class RooflineAnalyzer {
public:
    RooflineAnalyzer(const HardwareSpec& hw) : hw_(hw) {
        // Ridge point: where compute ceiling meets bandwidth ceiling
        // peakTFLOPS [TFLOPS] / memoryBandwidthGBs [GB/s] = FLOPs/Byte
        ridgePoint_ = (hw_.peakTFLOPS * 1e12) / (hw_.memoryBandwidthGBs * 1e9);
    }

    RooflineResult analyze(const Kernel& k) const {
        RooflineResult r;
        r.arithmeticIntensity = k.arithmeticIntensity();
        r.ridgePoint = ridgePoint_;

        // Memory-bandwidth bound: achievable = AI * BW
        double memoryBoundTFLOPS = r.arithmeticIntensity * hw_.memoryBandwidthGBs * 1e9 / 1e12;

        // Compute bound: limited by peak TFLOPS
        double computeBoundTFLOPS = hw_.peakTFLOPS;

        r.achievableTFLOPS = std::min(memoryBoundTFLOPS, computeBoundTFLOPS);
        r.isComputeBound = (r.arithmeticIntensity >= ridgePoint_);
        r.utilizationPercent = (r.achievableTFLOPS / hw_.peakTFLOPS) * 100.0;

        return r;
    }

    void printRoofline(const std::vector<Kernel>& kernels) const {
        std::cout << "\n══════════════════════════════════════════════════════════════\n";
        std::cout << "Roofline Model: " << hw_.name << "\n";
        std::cout << "  Peak Compute:      " << hw_.peakTFLOPS << " TFLOPS\n";
        std::cout << "  Memory Bandwidth:  " << hw_.memoryBandwidthGBs << " GB/s\n";
        std::cout << "  Ridge Point:       " << ridgePoint_ << " FLOPs/Byte\n";
        std::cout << "  On-chip SRAM:      " << hw_.onChipSRAM_MB << " MB\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << std::left
                  << std::setw(25) << "Kernel"
                  << std::setw(18) << "Arith. Intensity"
                  << std::setw(18) << "Achievable TFLOPS"
                  << std::setw(15) << "Utilization"
                  << std::setw(18) << "Bound Type"
                  << "\n";
        std::cout << std::string(94, '-') << "\n";

        for (const auto& k : kernels) {
            auto r = analyze(k);
            std::cout << std::left
                      << std::setw(25) << k.name
                      << std::setw(18) << std::fixed << std::setprecision(2)
                      << r.arithmeticIntensity
                      << std::setw(18) << std::setprecision(4) << r.achievableTFLOPS
                      << std::setw(15) << std::setprecision(1) << r.utilizationPercent << "%"
                      << std::setw(18) << (r.isComputeBound ? "Compute-bound" : "Memory-bound")
                      << "\n";
        }
        std::cout << std::endl;
    }

private:
    HardwareSpec hw_;
    double ridgePoint_;
};

// Analyze energy-efficiency tradeoffs of specialization
void analyzeEnergyEfficiency() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Energy Efficiency of Specialized Hardware\n";
    std::cout << "(Relative to high-quality C code on CPU, compute-bound)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct Platform {
        std::string name;
        double perfPerWattMultiplier;
        double designCost_M;
        std::string programmability;
    };

    std::vector<Platform> platforms = {
        {"Energy-optimized CPU",      1.0,    0.0,    "Easiest"},
        {"Throughput GPU",            10.0,   0.0,    "Moderate (CUDA)"},
        {"Programmable DSP",          20.0,   1.0,    "Limited domain"},
        {"Domain-specific Accel.",    50.0,   5.0,    "DSLs (e.g., DNN)"},
        {"FPGA / Reconfigurable",     100.0,  10.0,   "Difficult (Verilog)"},
        {"Fixed-function ASIC",       1000.0, 100.0,  "Not programmable"},
    };

    std::cout << std::left
              << std::setw(28) << "Platform"
              << std::setw(20) << "Perf/Watt vs CPU"
              << std::setw(18) << "Design Cost"
              << "Programmability\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& p : platforms) {
        std::cout << std::left
                  << std::setw(28) << p.name
                  << std::setw(20) << (std::to_string((int)p.perfPerWattMultiplier) + "x")
                  << std::setw(18) << ("$" + std::to_string((int)p.designCost_M) + "M")
                  << p.programmability << "\n";
    }
    std::cout << std::endl;
}

// Analyze data movement energy cost (from lecture)
void analyzeDataMovementEnergy() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Data Movement Energy Cost (per operation, approximate)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct EnergyCost {
        std::string operation;
        double energy_pJ;
    };

    std::vector<EnergyCost> costs = {
        {"Integer op",              1.0},
        {"FP32 op",                 20.0},
        {"Read 64b from local SRAM", 26.0},
        {"Read 64b from LPDDR",     1200.0},
    };

    std::cout << std::left
              << std::setw(35) << "Operation"
              << "Energy (pJ)\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& c : costs) {
        std::cout << std::left << std::setw(35) << c.operation << c.energy_pJ << " pJ\n";
    }
    std::cout << "\nKey insight: recomputing values is cheaper than storing + reloading!\n";
    std::cout << "SRAM access: 26 pJ, LPDDR access: 1200 pJ (46x difference)\n\n";
}

// Analyze instruction overhead amortization
void analyzeInstructionOverhead() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Instruction Overhead Amortization\n";
    std::cout << "(Overhead of programmability relative to useful compute)\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct InstrType {
        std::string name;
        double overheadPercent;
    };

    std::vector<InstrType> instrs = {
        {"Half-precision FMA (1 op)",       2000.0},
        {"Half-precision DP4 (4 ops)",      500.0},
        {"Half-precision 4x4 MMA (256 ops)", 27.0},
    };

    std::cout << std::left
              << std::setw(40) << "Instruction Type"
              << "Control Overhead\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& i : instrs) {
        std::cout << std::left << std::setw(40) << i.name
                  << i.overheadPercent << "%\n";
    }
    std::cout << "\nPrinciple: amortize instruction stream cost across many operations\n";
    std::cout << "of a single complex instruction.\n\n";
}

int main() {
    std::cout << "=== Lecture 10: Roofline Model & Arithmetic Intensity ===\n";
    std::cout << "Stanford CS149 - Hardware Specialization\n\n";

    // Define hardware platforms
    HardwareSpec cpu = {
        "CPU (2 GHz, 8 cores, AVX-512)",
        0.5,    // 0.5 TFLOPS
        50.0,   // 50 GB/s
        30.0,   // 30 MB L3 cache
        1.0     // baseline
    };

    HardwareSpec gpu = {
        "NVIDIA H100 GPU",
        67.0,   // 67 TFLOPS fp32 (SIMD); 989 TFLOPS in tensor cores
        3350.0, // 3.35 TB/s HBM3
        50.0,   // 50 MB L2
        10.0
    };

    HardwareSpec gpuTensor = {
        "NVIDIA H100 (Tensor Cores, fp16)",
        989.0,  // 989 TFLOPS in tensor cores
        3350.0, // 3.35 TB/s
        50.0,
        10.0
    };

    HardwareSpec tpu = {
        "Google TPU v1 (Systolic Array)",
        92.0,   // 92 TFLOPS (int8)
        30.0,   // 30 GB/s (to host)
        28.0,   // 28 MB on-chip
        80.0    // ~80x perf/watt vs CPU+GPU combo
    };

    // Matrix multiply: high arithmetic intensity
    // C = A(MxK) * B(KxN): FLOPs = 2*M*N*K, bytes = M*K + K*N + M*N
    int M = 4096, K = 4096, N = 4096;
    double flops_gemm = 2.0 * M * N * K;
    double bytes_gemm = (double)(M * K + K * N + M * N) * 4.0;  // fp32 = 4 bytes

    // Vector add: low arithmetic intensity
    int vecLen = 1 << 20;  // 1M elements
    double flops_vec = (double)vecLen;
    double bytes_vec = (double)vecLen * 3.0 * 4.0;  // 2 inputs + 1 output, fp32

    // Convolution: moderate arithmetic intensity
    int H = 224, W = 224, C_in = 3, C_out = 64, Kh = 3, Kw = 3;
    double flops_conv = 2.0 * H * W * C_in * C_out * Kh * Kw;
    double bytes_conv = (H * W * C_in + C_out * C_in * Kh * Kw + H * W * C_out) * 4.0;

    std::vector<Kernel> kernels = {
        {"GEMM 4096x4096x4096", flops_gemm, bytes_gemm},
        {"Vector Add (1M)",     flops_vec, bytes_vec},
        {"Conv 3x3 (224x224)",  flops_conv, bytes_conv},
        {"DNN Layer (typical)", 1e10, 2e8},        // ~50 FLOPs/byte
        {"Attention (seq=2048)", 8e9, 4e9},         // ~2 FLOPs/byte
    };

    // Analyze on each platform
    RooflineAnalyzer cpuAnalyzer(cpu);
    cpuAnalyzer.printRoofline(kernels);

    RooflineAnalyzer gpuAnalyzer(gpu);
    gpuAnalyzer.printRoofline(kernels);

    RooflineAnalyzer tensorAnalyzer(gpuTensor);
    tensorAnalyzer.printRoofline(kernels);

    // Energy and overhead analysis
    analyzeEnergyEfficiency();
    analyzeDataMovementEnergy();
    analyzeInstructionOverhead();

    // Summary
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. GEMM is compute-bound even on H100 (AI = " << flops_gemm/bytes_gemm << ")\n";
    std::cout << "2. Vector add is always memory-bound (AI = " << flops_vec/bytes_vec << ")\n";
    std::cout << "3. Tensor cores shift ridge point: need higher AI to be compute-bound\n";
    std::cout << "4. Systolic arrays eliminate instruction overhead → 27% overhead vs 2000%\n";
    std::cout << "5. Data movement dominates energy: ~46x more energy for DRAM vs SRAM\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
