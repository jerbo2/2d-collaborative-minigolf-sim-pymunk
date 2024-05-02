import numpy as np
from pathfind import dijkstra
import itertools


def dupe_step(points, times):
    """This gives the path duplicates so the simulation can keep up / so that the robot can keep up with the point it's tracking"""
    return [
        coords for group in zip(*itertools.repeat(points, times)) for coords in group
    ]


def interpolate_path(path, dt):

    interpolated_path = [path[0]]

    for i in range(len(path) - 1):
        current_coords = path[i]
        next_coords = path[i + 1]

        num_points = max(
            np.abs(int((next_coords[0] - current_coords[0]) / dt)),
            np.abs(int((next_coords[1] - current_coords[1]) / dt)),
        )

        interpolated_x = np.linspace(current_coords[0], next_coords[0], num_points)
        interpolated_y = np.linspace(current_coords[1], next_coords[1], num_points)

        interpolated_path.extend(np.column_stack((interpolated_x, interpolated_y)))

    return interpolated_path[1:]


# def cast(space, start, end):
#     hit_info = space.segment_query_first(start, end, 10, pymunk.ShapeFilter())
#     print(hit_info)


def non_holonomic_control_sim(
    robot_body, ball_body, obstacles, graph, path, L, K, dt, obs_graph
):
    max_linear_velocity = 200
    max_angular_velocity = 6

    theta = robot_body.angle  # Get the current orientation of the robot
    # Position of P (tracking point)
    x_P = robot_body.position.x + L * np.cos(theta)
    y_P = robot_body.position.y + L * np.sin(theta)

    # check if there's a dang obstacle in the way
    if len(path) == 0:
        for obs in obstacles:
            if obs.name == "obstacle":
                obs.cache_bb()
            dist_from_obs = obs.point_query((x_P, y_P)).distance
            if dist_from_obs < 10:
                if dist_from_obs < 0:
                    # point is inside which is bad
                    print("fixed point inside obstacle")
                    closest_point_x = min(list(obs.bb), key=lambda x: abs(x - x_P))
                    closest_point_y = min(list(obs.bb), key=lambda y: abs(y - y_P))
                    x_P, y_P = closest_point_x, closest_point_y
                max_linear_velocity, max_angular_velocity = 0, 0
                try:
                    path, _ = dijkstra(
                        graph,
                        (int(x_P // 10), int(y_P // 10)),
                        (
                            int(ball_body.position.x // 10),
                            int(ball_body.position.y // 10),
                        ),
                    )

                except KeyError:
                    # print("Line is outside the grid")
                    break

                # scale path to world coordinates
                path = [(x * 10, y * 10) for x, y in path]
                # interpolate in between
                path = interpolate_path(path, dt)

                break

        # Desired position is the current ball position
        x_d, y_d = ball_body.position.x, ball_body.position.y

    else:
        x_d, y_d = path.pop(0)
        max_linear_velocity, max_angular_velocity = 100, 3

    e = np.array([x_d - x_P, y_d - y_P])

    # Derived form of matrix M
    M = np.array(
        [[np.cos(theta), -L * np.sin(theta)], [np.sin(theta), L * np.cos(theta)]]
    )

    # Inverting M
    M_inv = np.linalg.inv(M)

    # Control input
    u = np.dot(M_inv, np.dot(K, e))

    # Update velocities
    v = u[0]
    w = u[1]

    v = np.clip(v, -max_linear_velocity, max_linear_velocity)
    w = np.clip(w, -max_angular_velocity, max_angular_velocity)

    # Update robot states
    robot_body.velocity = (v * np.cos(theta), v * np.sin(theta))
    robot_body.angular_velocity = w

    # Validate non-holonomic constraint
    if (
        -np.sin(theta) * robot_body.velocity[0] + np.cos(theta) * robot_body.velocity[1]
        > 1e-6
    ):
        print("Non-holonomic constraint violated")

    return x_P, y_P, path
