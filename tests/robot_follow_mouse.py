import pygame
import pymunk
import pymunk.pygame_util
import sys, os
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from pathfind import create_grid
from control import non_holonomic_control_sim
from drawing import CustomDrawOptions
from robot_shot_helpers import *

COLLTYPE_ROBOT = 0b0001


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


def main():
    pygame.init()
    width, height = 600, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Minigolf Simulation")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    space.gravity = (0, 0)

    L = 20  # Distance from robot's center to tracking point P
    K = np.array(
        [[10, 0], [0, 10]]
    )  # Simple proportional control gain for demonstration
    robot_body = create_robot(space, (500, 199))

    draw_options = CustomDrawOptions(screen)

    ghost_ball_shape_2 = add_ghost_ball(space, (0, 0), (255, 255, 0, 100))
    ghost_ball_shape_3 = add_ghost_ball(space, (0, 0), (0, 191, 255, 100))

    obstacles = []
    obstacles_graph = {}

    graph = create_grid(width, height, 10, obstacles_graph)
    path = []

    running = True

    body_to_track = ghost_ball_shape_2.body

    dt = 1
    obstacles_to_use = obstacles

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                body_to_track.position = pos

        x_P, y_P, path = non_holonomic_control_sim(
            robot_body, body_to_track, obstacles_to_use, graph, path, L, K, dt
        )

        ghost_ball_shape_3.body.position = (x_P, y_P)

        screen.fill((0, 128, 0))  # Green background

        space.debug_draw(draw_options)
        space.step(1 / 120.0)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
