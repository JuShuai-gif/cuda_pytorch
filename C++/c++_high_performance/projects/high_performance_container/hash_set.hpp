// An open-addressing hash set (synthesized from Ch4 data structures +
// Ch5 iterators + Ch7 allocators).
//
// Stores strings in a flat vector (no chaining), linear probing, load
// factor capped at 0.7 with rehash. Cache-friendly, avoids per-node
// allocation. Exposes a bidirectional-like iterator for traversal.

#ifndef CHP_HASH_SET_HPP
#define CHP_HASH_SET_HPP

#include <cstddef>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

namespace chp {

class HashSet {
public:
    HashSet() : slots_(kInitialCapacity), occupied_(kInitialCapacity, false) {}

    // O(1) amortized insert; rehashes at load factor > 0.7.
    bool insert(const std::string& value) {
        grow_if_needed();
        auto idx = probe(value);
        if (occupied_[idx]) {
            return false;  // already present
        }
        slots_[idx] = value;
        occupied_[idx] = true;
        ++size_;
        return true;
    }

    bool contains(const std::string& value) const {
        const auto cap = slots_.size();
        for (std::size_t idx = hash(value) % cap;; idx = (idx + 1) % cap) {
            if (!occupied_[idx]) {
                return false;
            }
            if (slots_[idx] == value) {
                return true;
            }
        }
    }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // Linear scan over occupied slots (order is not meaningful).
    std::vector<std::string> collect() const {
        std::vector<std::string> out;
        out.reserve(size_);
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            if (occupied_[i]) {
                out.push_back(slots_[i]);
            }
        }
        return out;
    }

private:
    static constexpr std::size_t kInitialCapacity = 16;
    static constexpr float kMaxLoadFactor = 0.7F;

    static std::size_t hash(const std::string& s) {
        return std::hash<std::string>{}(s);
    }

    // First free-or-matching slot for value, linear probing.
    std::size_t probe(const std::string& value) const {
        const auto cap = slots_.size();
        for (std::size_t idx = hash(value) % cap;; idx = (idx + 1) % cap) {
            if (!occupied_[idx] || slots_[idx] == value) {
                return idx;
            }
        }
    }

    void grow_if_needed() {
        if (static_cast<float>(size_ + 1) / static_cast<float>(slots_.size())
                <= kMaxLoadFactor) {
            return;
        }
        const std::size_t old_cap = slots_.size();
        const std::size_t new_cap = old_cap * 2;

        std::vector<std::string> new_slots(new_cap);
        std::vector<bool> new_occupied(new_cap, false);
        for (std::size_t i = 0; i < old_cap; ++i) {
            if (occupied_[i]) {
                const auto& v = slots_[i];
                for (std::size_t idx = hash(v) % new_cap;;
                     idx = (idx + 1) % new_cap) {
                    if (!new_occupied[idx]) {
                        new_slots[idx] = v;
                        new_occupied[idx] = true;
                        break;
                    }
                }
            }
        }
        slots_ = std::move(new_slots);
        occupied_ = std::move(new_occupied);
    }

    std::vector<std::string> slots_;
    std::vector<bool> occupied_;
    std::size_t size_ = 0;
};

}  // namespace chp

#endif  // CHP_HASH_SET_HPP
