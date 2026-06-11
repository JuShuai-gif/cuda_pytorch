"""Small simulator for CUDA shared-memory bank mappings.

It does not model NVIDIA hardware perfectly. The goal is to make the address,
word, bank, and rough transaction grouping visible while reading the summary.
"""

from collections import defaultdict


WARP_SIZE = 32
WORDS_PER_BANK_SET = 32


def banks_for_vector(vector_index: int, words_per_vector: int) -> list[int]:
    start_word = vector_index * words_per_vector
    return [((start_word + k) % WORDS_PER_BANK_SET) for k in range(words_per_vector)]


def words_for_vector(vector_index: int, words_per_vector: int) -> list[int]:
    start_word = vector_index * words_per_vector
    return [start_word + k for k in range(words_per_vector)]


def has_bank_conflict(accesses: list[tuple[int, int]]) -> bool:
    """Return True if one bank serves multiple different words."""
    words_by_bank: dict[int, set[int]] = defaultdict(set)
    for word, bank in accesses:
        words_by_bank[bank].add(word)
    return any(len(words) > 1 for words in words_by_bank.values())


def print_group(name: str, tids: range, addr_fn, words_per_vector: int) -> None:
    accesses: list[tuple[int, int]] = []
    print(f"\n{name}")
    print("-" * len(name))
    for tid in tids:
        vec = addr_fn(tid)
        words = words_for_vector(vec, words_per_vector)
        banks = banks_for_vector(vec, words_per_vector)
        accesses.extend(zip(words, banks))
        print(f"t{tid:02d}: vec={vec:02d}, words={words}, banks={banks}")
    print(f"bank_conflict={has_bank_conflict(accesses)}")


def xor_merge_ok(addr_fn, peer_delta: int) -> bool:
    for tid in range(WARP_SIZE):
        peer = tid ^ peer_delta
        if addr_fn(tid) != addr_fn(peer):
            return False
    return True


def analyze_case(name: str, addr_fn, words_per_vector: int, group_size: int) -> None:
    print(f"\n=== {name} ===")
    print(f"words_per_thread={words_per_vector}")
    print(f"xor1_merge_condition={xor_merge_ok(addr_fn, 1)}")
    print(f"xor2_merge_condition={xor_merge_ok(addr_fn, 2)}")

    for start in range(0, WARP_SIZE, group_size):
        stop = start + group_size
        print_group(f"transaction-sized group t{start:02d}-t{stop - 1:02d}", range(start, stop), addr_fn, words_per_vector)


def main() -> None:
    cases = [
        ("uint2_contiguous", lambda tid: tid, 2, 16),
        ("uint2_pair_broadcast", lambda tid: tid // 2, 2, 16),
        ("uint4_contiguous", lambda tid: tid, 4, 8),
        ("uint4_pair_merge", lambda tid: (tid // 8) * 2 + ((tid % 8) // 2) % 2, 4, 8),
        ("uint4_conflict_like_pdf", lambda tid: (tid // 16) * 4 + (tid % 16) // 8 + (tid % 8) // 4 * 8, 4, 8),
    ]

    for name, addr_fn, words_per_vector, group_size in cases:
        analyze_case(name, addr_fn, words_per_vector, group_size)


if __name__ == "__main__":
    main()

