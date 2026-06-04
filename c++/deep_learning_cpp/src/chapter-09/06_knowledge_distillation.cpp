/*
 * 06_knowledge_distillation.cpp
 * Chapter 9: Transformers and LLM Fine-Tuning in C++
 *
 * Knowledge Distillation transfers knowledge from a large "teacher" model
 * to a smaller "student" model. The key insight is "dark knowledge" —
 * the teacher's soft output distribution contains rich information about
 * inter-class relationships that one-hot labels cannot capture.
 *
 * Process:
 *   1. Train a large teacher model
 *   2. Produce soft targets using high temperature T
 *      soft_target = softmax(teacher_logits / T)
 *   3. Train student with combined loss:
 *      L = alpha * L_soft * T^2 + (1 - alpha) * L_hard
 *      where:
 *        L_soft  = KL_div(student_soft || teacher_soft)
 *        L_hard  = CrossEntropy(student_logits, true_labels)
 *
 * The T^2 factor scales the soft loss to match the hard loss gradient
 * magnitude, since softmax temperature scaling compresses gradients.
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
// Simple MLP Teacher: larger, more capacity
// ----------------------------------------------------------------
struct TeacherMLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    TeacherMLP(int input_dim, int hidden_dim, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x); // logits
    }
};

// ----------------------------------------------------------------
// Simple MLP Student: smaller, fewer parameters
// ----------------------------------------------------------------
struct StudentMLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    StudentMLP(int input_dim, int hidden_dim, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x); // logits
    }
};

// ----------------------------------------------------------------
// Helper: generate synthetic classification data
// ----------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> generateData(int n_samples,
                                                     int input_dim,
                                                     int num_classes) {
    auto x = torch::randn({n_samples, input_dim});
    // Synthetic labels: assign classes based on feature clusters
    auto logits = x.slice(1, 0, num_classes);
    auto labels = logits.argmax(1);
    return {x, labels};
}

// ----------------------------------------------------------------
// Train teacher model
// ----------------------------------------------------------------
void trainTeacher(std::shared_ptr<TeacherMLP> teacher,
                  const torch::Tensor &x_train,
                  const torch::Tensor &y_train,
                  int epochs = 50) {
    auto optimizer = torch::optim::Adam(teacher->parameters(), 0.001);

    for (int epoch = 0; epoch < epochs; epoch++) {
        teacher->train();
        optimizer.zero_grad();

        auto logits = teacher->forward(x_train);
        auto loss = torch::nn::functional::cross_entropy(logits, y_train);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "  Teacher epoch " << (epoch + 1)
                      << "/" << epochs << " loss: "
                      << std::setprecision(4) << loss.item<float>() << "\n";
        }
    }
}

// ----------------------------------------------------------------
// Compute soft targets from teacher with temperature
//
// soft_target = softmax(teacher_logits / T)
//
// Higher T produces softer (more uniform) distribution,
// revealing "dark knowledge" about class relationships.
// ----------------------------------------------------------------
torch::Tensor computeSoftTargets(
    std::shared_ptr<TeacherMLP> teacher,
    const torch::Tensor &x,
    float temperature) {
    teacher->eval();
    torch::NoGradGuard no_grad;
    auto logits = teacher->forward(x);
    return torch::softmax(logits / temperature, /*dim=*/-1);
}

// ----------------------------------------------------------------
// Distillation loss: KL divergence between student and teacher softmax
//
// L_soft = T^2 * KL(softmax(student/T) || softmax(teacher/T))
//
// The T^2 multiplier compensates for the gradient scaling introduced
// by the temperature in the softmax.
// ----------------------------------------------------------------
torch::Tensor distillationLoss(
    const torch::Tensor &student_logits,
    const torch::Tensor &teacher_soft_targets,
    float temperature) {
    auto student_log_soft = torch::log_softmax(
        student_logits / temperature, /*dim=*/-1);
    auto teacher_soft = teacher_soft_targets;

    // KL divergence: sum(teacher * (log(teacher) - log(student)))
    // = -sum(teacher * log(student)) + const  (const dropped in optimization)
    auto kl = torch::nn::functional::kl_div(
        student_log_soft, teacher_soft,
        torch::nn::functional::KLDivFuncOptions()
            .reduction(torch::kBatchMean)
            .log_target(false));

    return kl * (temperature * temperature);
}

// ----------------------------------------------------------------
// Train student via knowledge distillation
// ----------------------------------------------------------------
void trainStudentWithDistillation(
    std::shared_ptr<StudentMLP> student,
    std::shared_ptr<TeacherMLP> teacher,
    const torch::Tensor &x_train,
    const torch::Tensor &y_train,
    float temperature = 4.0,
    float alpha = 0.7,
    int epochs = 100) {
    // Pre-compute teacher soft targets (teacher is frozen)
    auto teacher_soft = computeSoftTargets(teacher, x_train, temperature);

    auto optimizer = torch::optim::Adam(student->parameters(), 0.001);

    for (int epoch = 0; epoch < epochs; epoch++) {
        student->train();
        optimizer.zero_grad();

        auto logits = student->forward(x_train);

        // Hard loss: standard cross-entropy with ground truth
        auto hard_loss = torch::nn::functional::cross_entropy(logits, y_train);

        // Soft loss: KL divergence with teacher's soft targets
        auto soft_loss = distillationLoss(logits, teacher_soft, temperature);

        // Combined loss
        auto total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss;

        total_loss.backward();
        optimizer.step();

        if ((epoch + 1) % 20 == 0) {
            // Compute accuracy
            student->eval();
            torch::NoGradGuard no_grad;
            auto pred = student->forward(x_train).argmax(1);
            auto acc = pred.eq(y_train).to(torch::kFloat32).mean();

            std::cout << "  Student epoch " << (epoch + 1) << "/" << epochs
                      << " total_loss: " << std::setprecision(4) << total_loss.item<float>()
                      << " acc: " << acc.item<float>()
                      << " hard: " << hard_loss.item<float>()
                      << " soft: " << soft_loss.item<float>() << "\n";
        }
    }
}

// ----------------------------------------------------------------
// Baseline: train student directly without distillation
// ----------------------------------------------------------------
void trainStudentBaseline(
    std::shared_ptr<StudentMLP> student,
    const torch::Tensor &x_train,
    const torch::Tensor &y_train,
    int epochs = 100) {
    auto optimizer = torch::optim::Adam(student->parameters(), 0.001);

    for (int epoch = 0; epoch < epochs; epoch++) {
        student->train();
        optimizer.zero_grad();

        auto logits = student->forward(x_train);
        auto loss = torch::nn::functional::cross_entropy(logits, y_train);
        loss.backward();
        optimizer.step();
    }
}

// ----------------------------------------------------------------
// Demo: Compare student with vs without distillation
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Knowledge Distillation Demo ===\n\n";

    int input_dim = 16;
    int num_classes = 5;
    int n_train = 200;
    int n_test = 50;
    float temperature = 4.0;
    float alpha = 0.7;

    auto [x_train, y_train] = generateData(n_train, input_dim, num_classes);
    auto [x_test, y_test] = generateData(n_test, input_dim, num_classes);

    // Model sizes
    auto teacher = std::make_shared<TeacherMLP>(input_dim, 64, num_classes);
    auto student_distilled = std::make_shared<StudentMLP>(input_dim, 16, num_classes);
    auto student_baseline = std::make_shared<StudentMLP>(input_dim, 16, num_classes);

    // Copy initial weights for fair comparison
    {
        auto params_dist = student_distilled->named_parameters();
        auto params_base = student_baseline->named_parameters();
        for (auto &item : params_base) {
            auto &p_base = item.value();
            auto &p_dist = params_dist[item.key()];
            p_base.data().copy_(p_dist.data());
        }
    }

    // Count parameters
    int64_t teacher_params = 0, student_params = 0;
    for (const auto &p : teacher->parameters()) teacher_params += p.numel();
    for (const auto &p : student_distilled->parameters()) student_params += p.numel();

    std::cout << "Model sizes:\n";
    std::cout << "  Teacher: " << teacher_params << " params\n";
    std::cout << "  Student: " << student_params << " params"
              << " (" << (100.0 * student_params / teacher_params)
              << "% of teacher)\n\n";

    // Train teacher
    std::cout << "Step 1: Train teacher\n";
    trainTeacher(teacher, x_train, y_train);
    {
        torch::NoGradGuard no_grad;
        teacher->eval();
        auto pred = teacher->forward(x_test).argmax(1);
        auto acc = pred.eq(y_test).to(torch::kFloat32).mean();
        std::cout << "  Teacher test accuracy: " << acc.item<float>() << "\n\n";
    }

    // Train student with distillation
    std::cout << "Step 2: Train student WITH distillation (T="
              << temperature << ", alpha=" << alpha << ")\n";
    trainStudentWithDistillation(student_distilled, teacher,
                                 x_train, y_train, temperature, alpha);
    {
        torch::NoGradGuard no_grad;
        student_distilled->eval();
        auto pred = student_distilled->forward(x_test).argmax(1);
        auto acc = pred.eq(y_test).to(torch::kFloat32).mean();
        std::cout << "  Student (distilled) test accuracy: "
                  << acc.item<float>() << "\n\n";
    }

    // Train student baseline (without distillation)
    std::cout << "Step 3: Train student WITHOUT distillation (baseline)\n";
    trainStudentBaseline(student_baseline, x_train, y_train);
    {
        torch::NoGradGuard no_grad;
        student_baseline->eval();
        auto pred = student_baseline->forward(x_test).argmax(1);
        auto acc = pred.eq(y_test).to(torch::kFloat32).mean();
        std::cout << "  Student (baseline) test accuracy: "
                  << acc.item<float>() << "\n\n";
    }

    std::cout << "--- Summary ---\n";
    std::cout << "Distillation transfers the teacher's learned patterns\n";
    std::cout << "to a smaller student, often outperforming direct training.\n";
    std::cout << "The soft targets carry 'dark knowledge' about class similarity.\n";

    return 0;
}
