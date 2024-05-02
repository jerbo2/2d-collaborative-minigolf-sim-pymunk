import numpy as np
import math, pymunk


def adjust_for_obstacles(obstacles, body_to_track, current_angle):
    for obs in obstacles:
        if obs.point_query(body_to_track.position).distance < 10:
            print("bad angle of attack")
            direction = (obs.body.position - body_to_track.position).normalized()
            new_angle = np.degrees(np.arctan2(direction.y, direction.x))
            print(f"new best angle: {new_angle}")
            return new_angle
    return current_angle


def evaluate_shots(shot_options, hole_position, idle_pos):
    shots = {}
    for test_ball, direction in shot_options:
        distance = (test_ball.body.position - hole_position).length
        if test_ball.body.position == idle_pos:
            print(f"found a direct shot at {direction} degrees")
            shots[direction] = 0
        else:
            shots[direction] = distance
    return {k: v for k, v in sorted(shots.items(), key=lambda item: item[1])}


def reset_balls(test_shot_ball_shapes, idle_pos):
    for test_ball in test_shot_ball_shapes:
        test_ball.body.velocity = (0, 0)
        test_ball.body.position = idle_pos


def calculate_shot_position(start_pos, angle_degrees, distance):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Calculate the new position
    new_x = start_pos[0] + distance * math.cos(angle_radians)
    new_y = start_pos[1] + distance * math.sin(angle_radians)

    return pymunk.Vec2d(new_x, new_y)


def simulate_shot_options(ghost_balls, directions, powers):
    pairs = []
    for ghost_ball, direction, power in zip(ghost_balls, directions, powers):
        radians = np.deg2rad(direction)
        initial_velocity = pymunk.Vec2d(
            np.cos(radians) * power, np.sin(radians) * power
        )
        ghost_ball.body.velocity = initial_velocity
        pairs.append((ghost_ball, direction))

    return pairs
