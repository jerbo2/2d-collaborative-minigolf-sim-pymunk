import pymunk
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.pygame_util import to_pygame
from typing import Sequence, Tuple


class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, screen):
        super().__init__(screen)

    def draw_circle(
        self,
        pos: pymunk.Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        if fill_color != SpaceDebugColor(0, 128, 0, 0):
            super().draw_circle(pos, angle, radius, outline_color, fill_color)

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        if fill_color != SpaceDebugColor(0, 128, 0, 0):
            super().draw_polygon(verts, radius, outline_color, fill_color)
