"""
Prefix caching with Radix tree for KV cache reuse.

Stores KV cache blocks keyed by token sequences in a radix tree
(compact prefix tree). New requests that share a prefix with
previously processed requests can skip the prefill phase entirely
by reusing cached KV blocks via copy-on-write.

This is the core technique behind SGLang's RadixAttention and
vLLM's automatic prefix caching, saving 50-70% of prefill
computation for workloads with shared system prompts.

Key features:
  - RadixTree: Compact trie with path compression for token sequences
  - Prefix match: Find longest common prefix for incoming requests
  - Copy-on-write: Shared KV cache blocks for matched prefixes
  - LRU eviction: Evict least recently used prefixes under memory pressure
  - Statistics: Hit rate, bytes saved, prefill time saved

Reference:
  Zheng et al., "SGLang: Efficient Execution of Structured Language
  Model Programs", NeurIPS 2024.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
#  Radix tree node
# ==============================================================================


@dataclass
class RadixNode:
    """A node in the radix tree representing a token subsequence.

    Unlike a standard trie where each node represents a single token,
    radix tree nodes may represent a sequence of tokens (path compression).
    This reduces tree depth and memory overhead.

    Attributes:
        token_seq: The token subsequence this node represents.
        children: Child nodes indexed by their first token.
        seq_id: Sequence ID that owns the KV cache for this path.
        block_table: Reference to the KV cache block table for this prefix.
        kv_len: Number of KV cache tokens stored at this node.
        last_access_time: Timestamp of last access (for LRU eviction).
        ref_count: Number of active sequences referencing this node.
    """

    token_seq: tuple[int, ...] = field(default_factory=tuple)
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    seq_id: int = -1  # -1 means no stored KV cache
    block_table: object | None = None  # Reference to KV cache blocks
    kv_len: int = 0
    last_access_time: float = 0.0
    ref_count: int = 1


# ==============================================================================
#  Radix tree
# ==============================================================================


class RadixTree:
    """Radix tree for storing and matching token sequences.

    Stores KV cache blocks indexed by token prefixes. The tree uses
    path compression: a node can represent multiple consecutive tokens
    as long as there is no branching point.

    Usage:
        tree = RadixTree()
        tree.insert([1, 2, 3], seq_id=42)
        match_len, node = tree.match([1, 2, 3, 4])
        # match_len = 3, node = the node holding tokens (1,2,3)

    Args:
        max_nodes: Maximum number of nodes before eviction triggers.
    """

    def __init__(self, max_nodes: int = 10000) -> None:
        self.root = RadixNode(token_seq=(), kv_len=0)
        self.max_nodes = max_nodes
        self._node_count = 1  # Root
        self._hit_count = 0
        self._miss_count = 0
        self._bytes_saved = 0
        self._prefill_time_saved_s = 0.0

    def insert(
        self,
        tokens: list[int],
        seq_id: int,
        block_table: object | None = None,
        kv_len: int | None = None,
    ) -> RadixNode:
        """Insert a token sequence into the radix tree.

        Creates necessary nodes with path compression. If a prefix
        already exists, only the new suffix is added.

        Args:
            tokens: List of token IDs forming the sequence.
            seq_id: Sequence ID associated with this path.
            block_table: KV cache block table to store.
            kv_len: Total KV cache tokens at leaf node.

        Returns:
            The leaf node containing the full sequence.
        """
        return self._insert_recursive(
            self.root,
            tuple(tokens),
            seq_id,
            block_table,
            kv_len if kv_len is not None else len(tokens),
        )

    def _insert_recursive(
        self,
        node: RadixNode,
        remaining: tuple[int, ...],
        seq_id: int,
        block_table: object | None,
        kv_len: int,
    ) -> RadixNode:
        """Recursively insert tokens, handling path compression and splits."""
        if not remaining:
            node.seq_id = seq_id
            node.block_table = block_table
            node.kv_len = kv_len
            node.last_access_time = time.monotonic()
            return node

        first_token = remaining[0]

        if first_token in node.children:
            child = node.children[first_token]
            child.last_access_time = time.monotonic()
            child.ref_count += 1

            # Find common prefix between child's token_seq and remaining
            common_len = self._common_prefix_len(child.token_seq, remaining)
            child_seq = child.token_seq

            if common_len == len(child_seq):
                # Entire child is a prefix; descend into it
                return self._insert_recursive(
                    child,
                    remaining[common_len:],
                    seq_id,
                    block_table,
                    kv_len,
                )
            else:
                # Partial match: split the child node
                common = child_seq[:common_len]
                child_suffix = child_seq[common_len:]
                remaining_suffix = remaining[common_len:]

                # Create a new intermediate node for the common prefix
                intermediate = RadixNode(
                    token_seq=common,
                    kv_len=child.kv_len,
                    last_access_time=time.monotonic(),
                    ref_count=child.ref_count + 1,
                )
                self._node_count += 1

                # Update the child to represent its suffix
                child.token_seq = child_suffix

                # Attach restructured nodes
                node.children[first_token] = intermediate
                intermediate.children[child_suffix[0]] = child

                if not remaining_suffix:
                    intermediate.seq_id = seq_id
                    intermediate.block_table = block_table
                    intermediate.kv_len = kv_len
                    return intermediate
                else:
                    return self._insert_recursive(
                        intermediate,
                        remaining_suffix,
                        seq_id,
                        block_table,
                        kv_len,
                    )
        else:
            # No existing child; create a new leaf
            new_node = RadixNode(
                token_seq=remaining,
                seq_id=seq_id,
                block_table=block_table,
                kv_len=kv_len,
                last_access_time=time.monotonic(),
            )
            node.children[first_token] = new_node
            self._node_count += 1
            return new_node

    def match(self, tokens: list[int]) -> tuple[int, Optional[RadixNode]]:
        """Find the longest prefix match in the tree.

        Args:
            tokens: Token sequence to match against.

        Returns:
            Tuple of (matched_token_count, matched_node).
            node is None if no match found (not even root).
        """
        return self._match_recursive(self.root, tuple(tokens), 0, None)

    def _match_recursive(
        self,
        node: RadixNode,
        remaining: tuple[int, ...],
        matched_len: int,
        best_node: Optional[RadixNode],
    ) -> tuple[int, Optional[RadixNode]]:
        """Recursively find the deepest matching node."""
        current_best_node = best_node
        current_best_len = matched_len

        # Update best match if this node stores KV cache
        if node.seq_id >= 0:
            current_best_node = node
            current_best_len = matched_len

        if not remaining:
            self._hit_count += 1
            if current_best_node:
                current_best_node.last_access_time = time.monotonic()
            return current_best_len, current_best_node

        first_token = remaining[0]
        if first_token in node.children:
            child = node.children[first_token]
            child_seq = child.token_seq
            common = self._common_prefix_len(child_seq, remaining)

            if common > 0:
                child.last_access_time = time.monotonic()
                child.ref_count += 1
                return self._match_recursive(
                    child,
                    remaining[common:],
                    matched_len + common,
                    current_best_node,
                )

        self._miss_count += 1
        return current_best_len, current_best_node

    def evict_lru(self, num_nodes: int = 100) -> int:
        """Evict the least recently used nodes from the tree.

        Uses a priority queue over leaf nodes sorted by last_access_time.
        Nodes with ref_count > 1 (shared) are skipped.

        Args:
            num_nodes: Target number of nodes to evict.

        Returns:
            Number of nodes actually evicted.
        """
        if self._node_count <= self.max_nodes:
            return 0

        # Collect evictable leaf nodes with their access times
        candidates: list[tuple[float, int, RadixNode]] = []
        self._collect_evictable(self.root, [], candidates)

        heapq.heapify(candidates)

        evicted = 0
        target = max(0, self._node_count - self.max_nodes)

        while evicted < target and evicted < num_nodes and candidates:
            _, _, node = heapq.heappop(candidates)
            if node.ref_count <= 1 and node.seq_id >= 0:
                node.seq_id = -1
                node.block_table = None
                node.kv_len = 0
                evicted += 1

        return evicted

    def _collect_evictable(
        self,
        node: RadixNode,
        path: list[RadixNode],
        candidates: list[tuple[float, int, RadixNode]],
    ) -> None:
        """Collect evictable nodes via DFS."""
        if node.seq_id >= 0 and node.block_table is not None:
            candidates.append((node.last_access_time, len(path), node))

        for child in node.children.values():
            self._collect_evictable(child, path + [node], candidates)

    def contains(self, tokens: list[int]) -> bool:
        """Check if exactly the given token sequence is stored.

        Args:
            tokens: Token sequence to check.

        Returns:
            True if the exact sequence exists in the tree.
        """
        match_len, node = self.match(tokens)
        return match_len == len(tokens) and node is not None

    # ---- Statistics ----

    @property
    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        return self._node_count

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (matched requests / total matches)."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    @property
    def bytes_saved(self) -> int:
        """Estimated bytes saved via cache hits."""
        return self._bytes_saved

    def record_savings(self, prefill_time_s: float, bytes_saved: int) -> None:
        """Record savings from a cache hit.

        Args:
            prefill_time_s: Prefill time that was saved.
            bytes_saved: KV cache bytes saved.
        """
        self._prefill_time_saved_s += prefill_time_s
        self._bytes_saved += bytes_saved

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        self._hit_count = 0
        self._miss_count = 0
        self._bytes_saved = 0
        self._prefill_time_saved_s = 0.0

    # ---- Utility ----

    @staticmethod
    def _common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """Length of common prefix between two token sequences."""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return i
        return min_len


# ==============================================================================
#  PrefixCache - high-level interface
# ==============================================================================


class PrefixCache:
    """High-level prefix caching interface for the inference engine.

    Integrates the radix tree with KV cache block management to
    provide automatic prefix sharing. When a new request arrives,
    the cache checks for matching prefixes and reuses KV cache blocks
    via copy-on-write, avoiding redundant prefill computation.

    Args:
        max_nodes: Maximum radix tree nodes.
        block_size: KV cache block size (for savings estimation).
        bytes_per_token: Estimated bytes per KV cache token.
    """

    def __init__(
        self,
        max_nodes: int = 10000,
        block_size: int = 16,
        bytes_per_token: int = 128
        * 8
        * 2
        * 2,  # 2 layers * 8 heads * 128 dim * 2 (K+V)
    ) -> None:
        self._tree = RadixTree(max_nodes=max_nodes)
        self._block_size = block_size
        self._bytes_per_token = bytes_per_token
        self._total_prefill_saved_s = 0.0

    def find_prefix(
        self, tokens: list[int]
    ) -> tuple[int, Optional[int], Optional[object]]:
        """Find the longest cached prefix for a token sequence.

        Args:
            tokens: Input token IDs.

        Returns:
            Tuple of (matched_length, cached_seq_id, kv_block_table).
            (0, None, None) if no cache hit.
        """
        match_len, node = self._tree.match(tokens)
        if node is None or match_len == 0:
            return 0, None, None

        bytes_saved = match_len * self._bytes_per_token
        self._tree.record_savings(prefill_time_s=0.0, bytes_saved=bytes_saved)

        return match_len, node.seq_id, node.block_table

    def insert(
        self,
        tokens: list[int],
        seq_id: int,
        kv_block_table: object,
        kv_len: int | None = None,
    ) -> None:
        """Insert a completed sequence into the cache.

        Args:
            tokens: Token IDs of the full sequence.
            seq_id: Sequence identifier.
            kv_block_table: KV cache block table for the sequence.
            kv_len: Optional explicit KV length.
        """
        self._tree.insert(
            tokens=tokens,
            seq_id=seq_id,
            block_table=kv_block_table,
            kv_len=kv_len if kv_len is not None else len(tokens),
        )

    def evict(self, num_nodes: int = 100) -> int:
        """Evict LRU nodes to free memory.

        Args:
            num_nodes: Maximum nodes to evict.

        Returns:
            Number of nodes evicted.
        """
        return self._tree.evict_lru(num_nodes=num_nodes)

    def contains(self, tokens: list[int]) -> bool:
        """Check if exactly the given sequence is cached."""
        return self._tree.contains(tokens)

    @property
    def hit_rate(self) -> float:
        """Prefix cache hit rate."""
        return self._tree.hit_rate

    @property
    def bytes_saved(self) -> int:
        """Total bytes saved via prefix caching."""
        return self._tree.bytes_saved

    @property
    def node_count(self) -> int:
        """Number of nodes in the radix tree."""
        return self._tree.node_count

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._tree.reset_stats()
        self._total_prefill_saved_s = 0.0


# ==============================================================================
#  Verification
# ==============================================================================


def test_radix_tree() -> None:
    """Basic correctness test for the radix tree."""
    tree = RadixTree()

    # Insert sequences
    tree.insert([1, 2, 3], seq_id=1)
    tree.insert([1, 2, 4], seq_id=2)
    tree.insert([1, 2, 3, 5, 6], seq_id=3)

    # Exact match
    assert tree.contains([1, 2, 3]), "Should contain exact sequence [1,2,3]"

    # Prefix match
    length, node = tree.match([1, 2, 3, 7, 8])
    assert length == 3, f"Expected match length 3, got {length}"
    assert node is not None and node.seq_id == 1

    # Partial prefix match
    length, node = tree.match([1, 2, 5])
    assert length == 2, f"Expected match length 2, got {length}"

    # No match
    length, node = tree.match([9, 10])
    assert length == 0 and node is None, "Should have no match"

    # Hit/miss tracking
    assert tree.hit_rate > 0, "Should have hits"

    print(f"  RadixTree: {tree.node_count} nodes, hit_rate={tree.hit_rate:.2%}")
    print("  RadixTree tests passed!")


if __name__ == "__main__":
    test_radix_tree()
