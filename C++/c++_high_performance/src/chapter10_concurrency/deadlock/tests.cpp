// Correctness checks: std::lock prevents deadlock and preserves balances.

#include <array>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "test_utils.hpp"

namespace {

struct Account {
    explicit Account(int balance = 0) : balance_(balance) {}
    int balance_ = 0;
    std::mutex m_{};
};

void transfer_money(Account& from, Account& to, int amount) {
    std::unique_lock<std::mutex> lock1{from.m_, std::defer_lock};
    std::unique_lock<std::mutex> lock2{to.m_, std::defer_lock};
    std::lock(lock1, lock2);
    from.balance_ -= amount;
    to.balance_ += amount;
}

}  // namespace

int main() {
    // 4 accounts, 8 transfers in mixed directions, including self-transfer.
    std::array<Account, 4> accounts{
        Account{1000}, Account{1000}, Account{1000}, Account{1000}};

    std::vector<std::thread> ts;
    const std::pair<int, int> routes[] = {{0, 1}, {1, 0}, {2, 3}, {3, 2},
                                          {0, 3}, {3, 1}, {1, 2}, {2, 0}};
    for (const auto& [from, to] : routes) {
        ts.emplace_back(transfer_money, std::ref(accounts[from]),
                        std::ref(accounts[to]), 10);
    }
    for (auto& t : ts) {
        t.join();
    }

    int total = 0;
    for (const auto& acc : accounts) {
        total += acc.balance_;
    }
    CHP_CHECK(total == 4000);  // conservation: no lost/created money

    return chp::test_summary("deadlock");
}
