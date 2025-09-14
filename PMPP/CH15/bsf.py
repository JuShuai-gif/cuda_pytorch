def dfs(scr_ptrs: list[int], dst: list[int], starting_node: int) -> list[int]:
    """we take a matrix in CRS format as an input and we return levels for all elements"""
    assert (
        0 <= starting_node < len(scr_ptrs)
    ), f"Staring node not among the {len(scr_ptrs)} nodes of input"

    ### init levels
    levels = [-1 for _ in scr_ptrs]
    already_visited = [False for _ in scr_ptrs]

    levels[starting_node] = 0
    already_visited[starting_node] = True

    queue = [starting_node]

    while queue:
        vertex = queue.pop(0)
        print("Vertex: ", vertex)
        # get all elements
        for i in range(scr_ptrs[vertex], scr_ptrs[vertex + 1]):
            neighbour = dst[i]
            if not already_visited[neighbour]:
                # if not yet visited, set the level as vertex+1 and mark as visited
                levels[neighbour] = levels[vertex] + 1
                already_visited[neighbour] = True
                queue.append(neighbour)

    return levels


if __name__ == "__main__":
    rowPtr = [0, 2, 5, 6, 8, 9, 11, 12, 15]
    colIdx = [2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6]
    values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    levels = dfs(rowPtr, colIdx, starting_node=0)

    for idx, _ in enumerate(rowPtr[:-1]):
        print(f"{idx}: level {levels[idx]}")
