// Avoiding deadlocks with std::lock (PDF p.294-295).
//
// transfer_money locks two accounts at the same time via std::lock on a
// pair of deferred unique_locks. Two threads transferring in opposite
// directions can never deadlock, and the total balance stays constant.

#include <cstdio>
#include <mutex>
#include <thread>

namespace {

struct Account {
    explicit Account(int balance) : balance_(balance) {}
    int balance_ = 0;
    std::mutex m_{};
};

void transfer_money(Account& from, Account& to, int amount) {
    std::unique_lock<std::mutex> lock1{from.m_, std::defer_lock};
    std::unique_lock<std::mutex> lock2{to.m_, std::defer_lock};
    std::lock(lock1, lock2);  // acquire both atomically, no ordering race
    from.balance_ -= amount;
    to.balance_ += amount;
}

}  // namespace

int main() {
    std::printf("== deadlock ==\n");

    Account a{1000};
    Account b{1000};

    // Opposite transfer directions: the classic deadlock setup. std::lock
    // guarantees no thread holds one lock while waiting for the other.
    std::thread t1{transfer_money, std::ref(a), std::ref(b), 100};
    std::thread t2{transfer_money, std::ref(b), std::ref(a), 50};
    t1.join();
    t2.join();

    const int total = a.balance_ + b.balance_;
    std::printf("a=%d b=%d total=%d\n", a.balance_, b.balance_, total);
    std::printf("total preserved: %s\n",
                total == 2000 ? "yes (no deadlock)" : "NO");

    return 0;
}
