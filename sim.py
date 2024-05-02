import pygame
import pymunk
import pymunk.pygame_util
import sys
import numpy as np
from copy import deepcopy

from pathfind import create_grid
from control import non_holonomic_control_sim
from drawing import CustomDrawOptions
from robot_shot_helpers import *

COLLTYPE_WALL = 0b0000
COLLTYPE_ROBOT = 0b0001
COLLTYPE_OBSTACLE = 0b0010
COLLTYPE_OBSTACLE_GUARD = 0b0011
COLLTYPE_BALL = 0b0100
COLLTYPE_HOLE = 0b1001
COLLTYPE_BALL_TEST = 0b0101
COLLTYPE_BALL_GUARD = 0b0110


def ignore_collision(arbiter, space, data):
    return False


def limit_ball_velocity(body, gravity, damping, dt):
    max_velocity = 500
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    l = body.velocity.length
    if l > max_velocity:
        scale = max_velocity / l
        body.velocity = body.velocity * scale


def add_ball(space, position, coll_type, radius=10, color=(255, 255, 255, 255)):
    """Add a ball to the space at a specified position"""
    mass = 1
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.7
    shape.friction = 0.5
    shape.color = color
    shape.collision_type = coll_type
    shape.filter = pymunk.ShapeFilter(categories=coll_type)
    shape.name = "ball"
    body.velocity_func = limit_ball_velocity
    space.add(body, shape)
    return shape


def add_hole(space, position, radius=15):
    hole_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    hole_body.position = position
    hole_shape = pymunk.Circle(hole_body, radius)
    hole_shape.collision_type = COLLTYPE_HOLE
    hole_shape.name = "hole"
    space.add(hole_body, hole_shape)
    return hole_shape


def add_walls(space, screen_width, screen_height):
    """Create walls around the screen"""
    walls = [
        pymunk.Segment(space.static_body, (0, 0), (0, screen_height), 1),
        pymunk.Segment(
            space.static_body, (0, screen_height), (screen_width, screen_height), 1
        ),
        pymunk.Segment(
            space.static_body, (screen_width, screen_height), (screen_width, 0), 1
        ),
        pymunk.Segment(space.static_body, (screen_width, 0), (0, 0), 1),
    ]
    for wall in walls:
        wall.elasticity = 0.8
        wall.friction = 0.5
        wall.collision_type = COLLTYPE_WALL
        wall.name = "wall"
        space.add(wall)


def add_obstacle(
    space, position, size=(50, 50), color=(255, 0, 0, 100), coll_type=COLLTYPE_OBSTACLE
):
    """Add an obstacle to the space at a specified position"""
    mass = 1e20
    inertia = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, inertia)
    body.position = position
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = 0.7
    shape.friction = 0.5
    shape.color = color
    shape.collision_type = coll_type
    shape.name = "obstacle"
    space.add(body, shape)
    return shape, body


def draw_arrow(screen, start, end, color=(255, 0, 0)):
    """Draw an arrow showing the direction and power of the shot"""
    pygame.draw.line(screen, color, start, end, 2)
    # draw arrow head
    angle = np.arctan2(start[1] - end[1], start[0] - end[0])
    pygame.draw.polygon(
        screen,
        color,
        (
            (
                end[0] - 5 * np.cos(angle + 3 * np.pi / 4),
                end[1] - 5 * np.sin(angle + 3 * np.pi / 4),
            ),
            (
                end[0] - 5 * np.cos(angle - 3 * np.pi / 4),
                end[1] - 5 * np.sin(angle - 3 * np.pi / 4),
            ),
            (end[0] - 10 * np.cos(angle), end[1] - 10 * np.sin(angle)),
        ),
    )


def create_robot(space, position):
    # Robot body
    robot_body = pymunk.Body(500, 1666, body_type=pymunk.Body.DYNAMIC)
    robot_body.position = position
    space.add(robot_body)

    robot_shape = pymunk.Poly.create_box(robot_body, (20, 20))
    robot_shape.color = (255, 0, 0, 100)  # Set the color of the robot to red
    robot_shape.collision_type = COLLTYPE_ROBOT
    robot_shape.name = "robot"
    robot_shape.elasticity = 1

    robot_shape.filter = pymunk.ShapeFilter(categories=COLLTYPE_ROBOT)
    space.add(robot_shape)

    return robot_body


def add_ghost_ball(space, position, color, radius=10):
    """Add a ghost ball to the space at a specified position."""
    mass = 1  # The mass is not really relevant since it won't interact physically.
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia, body_type=pymunk.Body.KINEMATIC)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.sensor = True  # Makes the shape a sensor.
    shape.color = color  # Set a semi-transparent blue color.
    space.add(body, shape)
    return shape


def update_ghost_ball_position_front(ghost_ball, robot_body, ball_body, power):
    """Position the ghost ball in line with the robot and the real ball."""
    direction = (ball_body.position - robot_body.position).normalized()
    distance = power * (ball_body.position - robot_body.position).length
    ghost_ball_position = ball_body.position + direction * distance
    ghost_ball.body.position = ghost_ball_position


def update_ghost_ball_position_behind(ghost_ball, hole, ball_body, distance):
    """Position the ghost ball behind the ball in line with the hole."""
    direction = (ball_body.position - hole).normalized()
    ghost_ball_position = ball_body.position + direction * distance
    ghost_ball.body.position = ghost_ball_position


def main():
    pygame.init()
    width, height = 600, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Minigolf Simulation")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0, 0)
    ball_start_pos = (150, 200)
    idle_pos = (600, 600)

    robots_turn = False
    aimed = False
    L = 20  # Distance from robot's center to tracking point P
    K = np.array(
        [[10, 0], [0, 10]]
    )  # Simple proportional control gain for demonstration
    robot_body = create_robot(space, (500, 199))

    draw_options = CustomDrawOptions(screen)
    hole_shape = add_hole(space, (500, 200), 15)  # Position the hole
    ghost_ball_shape = add_ghost_ball(space, (200, 200), (220, 20, 60, 100))
    ghost_ball_shape_2 = add_ghost_ball(space, (0, 0), (255, 255, 0, 100))
    ghost_ball_shape_3 = add_ghost_ball(space, (0, 0), (0, 191, 255, 100))
    test_shot_ball_shapes = [
        add_ball(space, idle_pos, COLLTYPE_BALL_TEST) for _ in range(72)
    ]
    ball_guard_shape = add_ball(
        space, ball_start_pos, COLLTYPE_BALL_GUARD, 30, color=(0, 128, 0, 0)
    )
    ball_shape = add_ball(space, ball_start_pos, COLLTYPE_BALL)
    add_walls(space, width, height)

    ### Add obstacles and create a graph for pathfinding

    obstacle_positions = [
        # (250, 150),
        # (250, 100),
        # (250, 200),
        # (250, 250),
        (300, 250),
        (300, 100),
        (350, 100),
        (350, 150),
        # (350, 200),
        # (350, 250),
        # (300, 300),
        # (300, 350),
    ]

    ignore_handlers = [
        space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_HOLE),
        space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_BALL_TEST),
        space.add_collision_handler(COLLTYPE_BALL_TEST, COLLTYPE_BALL_TEST),
        space.add_collision_handler(COLLTYPE_BALL_TEST, COLLTYPE_ROBOT),
        space.add_collision_handler(COLLTYPE_BALL_TEST, COLLTYPE_HOLE),
        space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_BALL_GUARD),
        space.add_collision_handler(COLLTYPE_ROBOT, COLLTYPE_WALL),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_BALL_TEST),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_BALL),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_ROBOT),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_WALL),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_HOLE),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_OBSTACLE),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_OBSTACLE_GUARD),
        space.add_collision_handler(COLLTYPE_OBSTACLE_GUARD, COLLTYPE_BALL_GUARD),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_BALL_TEST),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_BALL),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_ROBOT),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_WALL),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_HOLE),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_OBSTACLE),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_OBSTACLE_GUARD),
        space.add_collision_handler(COLLTYPE_BALL_GUARD, COLLTYPE_BALL_GUARD),
    ]

    for handler in ignore_handlers:
        handler.begin = ignore_collision

    obs_buffer = 0

    obstacles = []
    obstacles_graph = {}
    for pos in obstacle_positions:
        obstacle_shape, _ = add_obstacle(space, pos)
        obstacle_guard_shape, _ = add_obstacle(
            space,
            pos,
            size=(100, 100),
            color=(0, 128, 0, 0),
            coll_type=COLLTYPE_OBSTACLE_GUARD,
        )
        left, bottom, right, top = obstacle_guard_shape.bb
        for x in range(int(left // 10 - obs_buffer), int(right // 10 + obs_buffer), 1):
            for y in range(
                int(bottom // 10 - obs_buffer), int(top // 10 + obs_buffer), 1
            ):
                if obstacles_graph.get((x, y)) == None:
                    obstacles_graph[(x, y)] = True
        obstacles.append(obstacle_shape)
        obstacles.append(obstacle_guard_shape)

    obstacles_graph_dynamic = deepcopy(obstacles_graph)
    obstacles_graph_dynamic[
        ball_guard_shape.body.position.x // 10, ball_guard_shape.body.position.y // 10
    ] = True

    # add obstacle node to 2 layers of surrounding cells
    for x in range(-2, 3):
        for y in range(-2, 3):
            obstacles_graph_dynamic[
                (
                    ball_guard_shape.body.position.x // 10 + x,
                    ball_guard_shape.body.position.y // 10 + y,
                )
            ] = True

    obstacles_dynamic = deepcopy(obstacles)
    obstacles_dynamic.append(ball_guard_shape)

    graph = create_grid(width, height, 10, obstacles_graph_dynamic)
    path = []

    running = True
    aiming = False  # Flag to check if the user is aiming
    start_pos = (0, 0)

    # Initialize a timer at the beginning of your game or in relevant scope
    update_timer = 0
    update_interval = 2  # Interval in seconds
    current_step = 0

    body_to_track = ghost_ball_shape_2.body

    dt = 1

    current_step = 0
    best_angle = 0
    obstacles_to_use = obstacles_dynamic

    filter = pymunk.ShapeFilter(
        mask=pymunk.ShapeFilter.ALL_MASKS() ^ COLLTYPE_ROBOT ^ COLLTYPE_BALL_TEST
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                start_pos = pygame.mouse.get_pos()
                if ball_shape.point_query(start_pos).distance < 0:
                    aiming = True
                    graph = create_grid(width, height, 10, obstacles_graph)
                    obstacles_to_use = obstacles
            elif event.type == pygame.MOUSEBUTTONUP and aiming:
                aiming = False
                end_pos = pygame.mouse.get_pos()
                dx, dy = start_pos[0] - end_pos[0], start_pos[1] - end_pos[1]
                force = (dx * 5, dy * 5)
                ball_shape.body.apply_impulse_at_local_point(force)

        if ball_shape.body.velocity.length != 0:
            if (
                ball_shape.body.velocity.length < 5
                or distance_to_hole < hole_shape.radius
            ):
                # Stop the ball and reset various states
                ball_shape.body.velocity = (0, 0)
                ball_shape.body.angular_velocity = 0
                ball_shape.body.angle = 0

                # Reset game-related flags and counters
                robots_turn = not robots_turn
                L = 20
                aimed = False
                current_step = 0
                shots = {}

                # If the ball is close enough to the hole, reposition it to start
                if distance_to_hole < hole_shape.radius:
                    ball_shape.body.position = ball_start_pos
                    robots_turn = False

                # Update the position of the ball guard
                ball_guard_shape.body.position = ball_shape.body.position

                # Ensure the robot does not disturb the ball when it's not its turn
                obstacles_graph_dynamic = deepcopy(obstacles_graph)
                obstacles_to_use = obstacles_dynamic

                # Add the ball as an obstacle in the graph
                bx, by = (
                    ball_guard_shape.body.position.x // 10,
                    ball_guard_shape.body.position.y // 10,
                )
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        obstacles_graph_dynamic[(bx + x, by + y)] = True

                # Recreate the grid with updated obstacles
                graph = create_grid(width, height, 10, obstacles_graph_dynamic)

                for test_ball in test_shot_ball_shapes:
                    test_ball.body.position = ball_shape.body.position

            else:
                # Apply damping to the ball velocity
                ball_shape.body.velocity *= 0.99

            # The object to be tracked by the camera or UI
            body_to_track = ghost_ball_shape_2.body

        if not robots_turn:
            update_ghost_ball_position_front(
                ghost_ball_shape, robot_body, ball_shape.body, 2
            )
            update_ghost_ball_position_behind(
                ghost_ball_shape_2, hole_shape.body.position, ball_shape.body, 80
            )
            body_to_track = ghost_ball_shape_2.body

        update_timer += clock.get_time() / 1000.0

        if robots_turn:
            robot_calibration_ball_distance = (
                robot_body.position - body_to_track.position
            ).length

            if current_step == 0:
                for obs in obstacles:
                    print((ball_shape.body.position - obs.body.position).length)
                    if (ball_shape.body.position - obs.body.position).length < 55:
                        print("ball is too close to an obstacle")
                        robots_turn = False
                        current_step = 6
                        break

            if update_timer >= update_interval:
                if current_step in [3, 4]:
                    if not aimed:
                        print(f"aiming at {best_angle} degrees")
                        shot_power = 100
                        aimed = True
                    ahead_pos = np.clip(
                        calculate_shot_position(
                            ball_shape.body.position, best_angle, shot_power
                        ),
                        (0, 0),
                        (width, height),
                    )
                    ghost_ball_shape.body.position = tuple(ahead_pos)
                    # best_angle = adjust_for_obstacles(obstacles, body_to_track, best_angle)
                    update_timer = 0

                if current_step == 0:
                    print("giving balls a push")
                    shot_options = simulate_shot_options(
                        test_shot_ball_shapes,
                        np.arange(0, 360, 5),
                        np.linspace(200, 200, 72),
                    )
                    current_step = 1

                elif current_step == 1:
                    print("evaluating shots")
                    shots = evaluate_shots(
                        shot_options, hole_shape.body.position, idle_pos
                    )
                    angles = list(shots.keys())
                    best_angle = angles.pop(0)
                    current_step = 2

                elif current_step == 2:
                    print("resetting balls")
                    reset_balls(test_shot_ball_shapes, idle_pos)
                    current_step = 3

                elif current_step == 3:
                    print(
                        f"calibrating, current distance is {robot_calibration_ball_distance}"
                    )
                    behind_pos = np.clip(
                        calculate_shot_position(
                            ball_shape.body.position, best_angle + 180, shot_power
                        ),
                        (0, 0),
                        (width, height),
                    )
                    ghost_ball_shape_2.body.position = tuple(behind_pos)
                    L = 10  # Assuming L is used somewhere else to adjust position or logic

                    for obs in obstacles:
                        if (
                            obs.point_query(ghost_ball_shape_2.body.position).distance
                            < 10
                        ):
                            best_angle = angles.pop(0)
                            break

                    segment_query_info = space.segment_query_first(
                        ghost_ball_shape_2.body.position,
                        ball_shape.body.position,
                        10,
                        filter,
                    )

                    if segment_query_info.shape.name == "obstacle":
                        print(segment_query_info)
                        best_angle = angles.pop(0)

                    else:
                        current_step = 4 if robot_calibration_ball_distance < 20 else 3

                elif current_step == 4:
                    print("final positioning")
                    behind_pos = np.clip(
                        calculate_shot_position(
                            ball_shape.body.position, best_angle + 180, shot_power * 0.5
                        ),
                        (0, 0),
                        (width, height),
                    )
                    ghost_ball_shape_2.body.position = tuple(behind_pos)
                    obstacles_to_use = obstacles
                    graph = create_grid(width, height, 10, obstacles_graph)
                    current_step = 5

                elif current_step == 5:
                    print("shooting")
                    body_to_track = ghost_ball_shape.body
                    current_step = 6  # No further steps

                update_timer = 0  # Reset timer at the end of the routine

        # Check for ball in hole
        distance_to_hole = (ball_shape.body.position - hole_shape.body.position).length

        for test_ball in test_shot_ball_shapes:
            if (
                test_ball.body.position - hole_shape.body.position
            ).length < hole_shape.radius:
                test_ball.body.velocity = (0, 0)
                test_ball.body.angular_velocity = 0
                test_ball.body.position = idle_pos

        x_P, y_P, path = non_holonomic_control_sim(
            robot_body,
            body_to_track,
            obstacles_to_use,
            graph,
            path,
            L,
            K,
            dt,
            obstacles_graph_dynamic,
        )

        ghost_ball_shape_3.body.position = (x_P, y_P)

        screen.fill((0, 128, 0))  # Green background

        # draw arrow if aiming
        if aiming:
            mouse_pos = pygame.mouse.get_pos()
            # get distance from current mouse position to the ball
            dist_x = mouse_pos[0] - ball_shape.body.position.x
            dist_y = mouse_pos[1] - ball_shape.body.position.y
            # arrow will opposite direction of the mouse to show direction and power of the shot

            draw_arrow(
                screen,
                (ball_shape.body.position.x, ball_shape.body.position.y),
                (
                    ball_shape.body.position.x - dist_x,
                    ball_shape.body.position.y - dist_y,
                ),
            )

        space.debug_draw(draw_options)
        space.step(1 / 120.0)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
