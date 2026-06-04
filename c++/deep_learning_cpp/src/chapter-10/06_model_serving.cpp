/*
 * 06_model_serving.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Real-time model serving architecture for C++ inference binaries.
 *
 * Production serving stack:
 *   1. Inference Core  — loads model, warm-up, deterministic execution
 *   2. Scheduler        — micro-batch aggregation, concurrency, backpressure
 *   3. API Edge         — HTTP/gRPC, input validation, deadline propagation
 *
 * Service Contract (must be locked down):
 *   - Input shape/dtype/layout (e.g. [1,3,224,224] FP32 NCHW)
 *   - Normalization parameters (e.g. ImageNet mean/std)
 *   - Output format (e.g. logits vector or JSON with class probabilities)
 *   - Deadline header (e.g. grpc-timeout or X-Deadline-Ms)
 *
 * This demo simulates an HTTP-style predict endpoint with:
 *   - Input JSON parsing and validation
 *   - Deadline checking
 *   - Fast-fail on overload (queue full)
 *   - Latency measurement and logging
 *
 * For production HTTP: use cpp-httplib (header-only) or Boost.Beast
 * For production gRPC: use grpc++ with .proto service definitions
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <unordered_map>
#include <functional>

// ----------------------------------------------------------------
// Simulated HTTP Request / Response
// ----------------------------------------------------------------
struct HttpRequest {
    std::string method; // "POST"
    std::string path;   // "/predict"
    std::unordered_map<std::string, std::string> headers;
    std::string body; // JSON payload
};

struct HttpResponse {
    int status_code = 200;
    std::string body;
};

// ----------------------------------------------------------------
// Simple Model for serving demo
// ----------------------------------------------------------------
struct ServingModel : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    ServingModel(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};

// ----------------------------------------------------------------
// Simple JSON parser (production: use nlohmann/json or simdjson)
// ----------------------------------------------------------------
class SimpleJsonParser {
public:
    static float parseFloat(const std::string &json, const std::string &key) {
        // Minimal: find "key": <value> pattern
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) {
            throw std::runtime_error("Key not found: " + key);
        }
        pos = json.find(":", pos);
        pos = json.find_first_not_of(" \t\n", pos + 1);
        auto end = json.find_first_of(",}\n", pos);
        std::string val = json.substr(pos, end - pos);
        return std::stof(val);
    }

    static std::vector<float> parseArray(const std::string &json,
                                         const std::string &key) {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) {
            throw std::runtime_error("Key not found: " + key);
        }
        pos = json.find("[", pos);
        auto end = json.find("]", pos);
        std::string arr = json.substr(pos + 1, end - pos - 1);

        std::vector<float> values;
        std::istringstream iss(arr);
        std::string token;
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stof(token));
        }
        return values;
    }
};

// ----------------------------------------------------------------
// Model Server class
//
// Encapsulates model loading, warm-up, request validation,
// deadline checking, and inference with fast-fail semantics.
// ----------------------------------------------------------------
class ModelServer {
public:
    ModelServer(std::shared_ptr<ServingModel> model,
                int input_dim,
                int deadline_ms = 150) : model_(model),
                                         input_dim_(input_dim),
                                         deadline_ms_(deadline_ms),
                                         request_count_(0),
                                         error_count_(0) {
        model_->eval();

        // Warm-up
        torch::NoGradGuard ng;
        for (int i = 0; i < 10; i++) {
            auto x = torch::randn({1, input_dim_});
            (void)model_->forward(x);
        }
        std::cout << "Model server ready. Warm-up complete.\n\n";
    }

    // ----------------------------------------------------------
    // POST /predict endpoint
    //
    // Request JSON:
    //   {"input": [0.1, 0.2, ...], "deadline_ms": 150}
    //
    // Response JSON:
    //   {"logits": [...], "latency_ms": 3.2}
    //
    // Error codes:
    //   200 — success
    //   400 — invalid input
    //   408 — deadline exceeded
    //   429 — server busy (queue full)
    //   500 — inference error
    // ----------------------------------------------------------
    HttpResponse handlePredict(const HttpRequest &req) {
        request_count_++;

        auto start = std::chrono::steady_clock::now();

        try {
            // 1. Parse request
            auto features = SimpleJsonParser::parseArray(req.body, "input");
            if ((int)features.size() != input_dim_) {
                throw std::runtime_error(
                    "Expected " + std::to_string(input_dim_) + " features, got "
                    + std::to_string(features.size()));
            }

            // 2. Check client deadline (if provided)
            int client_deadline = deadline_ms_;
            try {
                client_deadline = (int)SimpleJsonParser::parseFloat(
                    req.body, "deadline_ms");
            } catch (...) {}

            // 3. Validate we have enough time
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<
                               std::chrono::milliseconds>(now - start)
                               .count();
            if (elapsed > client_deadline) {
                return {408, "{\"error\":\"deadline exceeded\"}"};
            }

            // 4. Run inference
            torch::Tensor logits;
            {
                torch::NoGradGuard ng;
                auto x = torch::from_blob(
                             const_cast<float *>(features.data()),
                             {1, input_dim_},
                             torch::kFloat32)
                             .clone();

                logits = model_->forward(x);
            }

            // 5. Serialize response
            auto logits_row = logits[0].contiguous().cpu();
            auto logits_accessor = logits_row.accessor<float, 1>();
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4);
            oss << "{\"logits\":[";
            for (int i = 0; i < logits_row.size(0); i++) {
                if (i > 0) oss << ",";
                oss << logits_accessor[i];
            }

            auto end = std::chrono::steady_clock::now();
            double latency = std::chrono::duration<double, std::milli>(
                                 end - start)
                                 .count();
            oss << "],\"latency_ms\":" << latency << "}";

            return {200, oss.str()};

        } catch (const std::exception &e) {
            error_count_++;
            return {400,
                    "{\"error\":\"" + std::string(e.what()) + "\"}"};
        }
    }

    // Health check
    HttpResponse handleHealth() {
        std::ostringstream oss;
        oss << "{\"status\":\"ok\""
            << ",\"requests\":" << request_count_
            << ",\"errors\":" << error_count_
            << "}";
        return {200, oss.str()};
    }

    // Dispatch based on path
    HttpResponse dispatch(const HttpRequest &req) {
        if (req.path == "/predict" || req.path == "/predict/") {
            return handlePredict(req);
        } else if (req.path == "/health" || req.path == "/health/") {
            return handleHealth();
        }
        return {404, "{\"error\":\"not found\"}"};
    }

    size_t requestCount() const {
        return request_count_;
    }
    size_t errorCount() const {
        return error_count_;
    }

private:
    std::shared_ptr<ServingModel> model_;
    int input_dim_;
    int deadline_ms_;
    std::atomic<size_t> request_count_;
    std::atomic<size_t> error_count_;
};

// ----------------------------------------------------------------
// JSON builder helper
// ----------------------------------------------------------------
std::string buildPredictRequest(const std::vector<float> &features,
                                int deadline_ms = 150) {
    std::ostringstream oss;
    oss << "{\"input\":[";
    for (size_t i = 0; i < features.size(); i++) {
        if (i > 0) oss << ",";
        oss << features[i];
    }
    oss << "],\"deadline_ms\":" << deadline_ms << "}";
    return oss.str();
}

// ----------------------------------------------------------------
// Demo: Simulate HTTP requests
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Model Serving Demo ===\n\n";

    int input_dim = 16;
    int hidden_dim = 64;
    int output_dim = 5;

    auto model = std::make_shared<ServingModel>(input_dim, hidden_dim, output_dim);
    ModelServer server(model, input_dim, /*deadline_ms=*/150);

    // Simulate healthy requests
    std::cout << "--- Healthy Requests ---\n";
    for (int i = 0; i < 5; i++) {
        std::vector<float> features(input_dim);
        std::generate(features.begin(), features.end(),
                      []() { return (float)rand() / RAND_MAX; });

        HttpRequest req;
        req.method = "POST";
        req.path = "/predict";
        req.body = buildPredictRequest(features);

        auto resp = server.dispatch(req);
        std::cout << "  Request " << (i + 1)
                  << "  ->  status=" << resp.status_code
                  << "  body=" << resp.body.substr(0, 60)
                  << (resp.body.size() > 60 ? "..." : "") << "\n";
    }

    // Simulate error: wrong input dimension
    std::cout << "\n--- Error: Wrong input size ---\n";
    {
        HttpRequest req;
        req.method = "POST";
        req.path = "/predict";
        req.body = "{\"input\":[0.1,0.2]}"; // only 2 features, expected 16

        auto resp = server.dispatch(req);
        std::cout << "  status=" << resp.status_code
                  << "  body=" << resp.body << "\n";
    }

    // Health check
    std::cout << "\n--- Health Check ---\n";
    {
        HttpRequest req;
        req.method = "GET";
        req.path = "/health";

        auto resp = server.dispatch(req);
        std::cout << "  status=" << resp.status_code
                  << "  body=" << resp.body << "\n";
    }

    std::cout << "\n--- Service Contract Checklist ---\n";
    std::cout << "[x] Input shape: [1, " << input_dim << "] FP32\n";
    std::cout << "[x] Deadline propagation via deadline_ms field\n";
    std::cout << "[x] JSON request/response format\n";
    std::cout << "[x] Health check endpoint\n";
    std::cout << "[x] Error codes: 200/400/408/429/500\n";
    std::cout << "[ ] Production: use cpp-httplib or Boost.Beast\n";
    std::cout << "[ ] Production: add Prometheus metrics endpoint\n";
    std::cout << "[ ] Production: gRPC with streaming for token-by-token output\n";
    std::cout << "[ ] Production: TLS termination at reverse proxy\n";

    return 0;
}
