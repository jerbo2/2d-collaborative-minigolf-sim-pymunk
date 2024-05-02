import heapq


def create_grid(width, height, cell_size, obstacles):
    # Convert dimensions to number of cells
    num_x_cells = (width // cell_size) + 1
    num_y_cells = (height // cell_size) + 1
    grid = {}

    # fig, ax = plt.subplots()

    for x in range(num_x_cells):
        for y in range(num_y_cells):
            grid_pos = (x, y)

            if (x, y) in obstacles:
                grid[grid_pos] = {}
                # ax.plot(x, y, 'ro')  # Plot obstacles in red
            else:
                grid[grid_pos] = {}
                # ax.plot(x, y, 'bo')  # Plot free nodes in blue
                # Connect to adjacent cells if not an obstacle
                if x > 0 and (x - 1, y) not in obstacles:
                    grid[grid_pos][(x - 1, y)] = 1  # Left
                if x < num_x_cells - 1 and (x + 1, y) not in obstacles:
                    grid[grid_pos][(x + 1, y)] = 1  # Right
                if y > 0 and (x, y - 1) not in obstacles:
                    grid[grid_pos][(x, y - 1)] = 1  # Up
                if y < num_y_cells - 1 and (x, y + 1) not in obstacles:
                    grid[grid_pos][(x, y + 1)] = 1  # Down

    # ax.grid(True)  # Enable grid lines
    # ax.set_aspect('equal', adjustable='datalim')  # Ensure the x and y axes are scaled equally
    # ax.invert_yaxis()  # Invert the y-axis
    # make y and x ticks larger
    # ax.tick_params(axis='both', which='major', labelsize=10)
    # plt.show()

    return grid


def dijkstra(graph, start, end):
    # Shortest paths is a dict of nodes whose value is a tuple of (previous node, weight)
    shortest_paths = {vertex: (None, float("infinity")) for vertex in graph}
    print(shortest_paths)
    shortest_paths[start] = (None, 0)

    # Priority queue to store vertices with their corresponding weights
    priority_queue = [(0, start)]

    while priority_queue:
        # print(priority_queue)
        # Get the vertex with the lowest weight
        current_weight, current_vertex = heapq.heappop(priority_queue)

        # Nodes can only be visited once, we only process a vertex the first time we remove it from the priority queue
        if current_vertex == end:
            break

        # Visit each neighbor of the current vertex
        for neighbor, weight in graph[current_vertex].items():
            distance = current_weight + weight
            # print(distance, neighbor, weight)

            # Only consider this new path if it's better than any path we've already found
            if distance < shortest_paths[neighbor][1]:
                # print(f"Node {neighbor} is closer to the start node ({current_vertex}) than before at a distance of {distance} where the previous distance was {shortest_paths[neighbor][1]}")
                shortest_paths[neighbor] = (current_vertex, distance)
                heapq.heappush(priority_queue, (distance, neighbor))

        # print(f"Shortest paths: {shortest_paths}")

    # Work backwards through paths to find the shortest path
    path, current_vertex = [], end
    while current_vertex is not None:
        path.insert(0, current_vertex)
        next_vertex = shortest_paths[current_vertex][0]
        current_vertex = next_vertex

    # Return the path and the weight of the end node
    return path, shortest_paths[end][1]


obstacles = {}
obstacles[(19, 19)] = 1
obstacles[(20, 19)] = 1
obstacles[(21, 19)] = 1


# graph = create_grid(600, 400, 10, obstacles)
# path, weight = dijkstra(graph, (20, 20), (20, 18))
# print(path, weight)

# plt.figure()
# plt.scatter(*zip(*graph.keys()))
# #plt.scatter(*zip(*obstacles.keys()), c='red')
# plt.plot(*zip(*path), c='green', lw=5)
# plt.show()
