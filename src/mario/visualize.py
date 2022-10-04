"""Tools for visualizing Mario levels.

Thanks to Ahmed Khalifa for providing this code.

Sprites for the level visualization are located in `src/mario/sprites`.

Usage:
    # In Python.
    from src.mario.visualize import visualize_level

    # Example run.
    python -m src.mario.visualize
"""
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src.mario.level import MarioLevel

os.environ['CLASSPATH'] = str(Path(__file__).parent / "Mario.jar")
from jnius import \
    autoclass  # pylint: disable = wrong-import-order, wrong-import-position, import-outside-toplevel

MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

# There are a lot of Java-style names in this file.
# pylint: disable = invalid-name


def getMarioGraphics():
    graphics = {
        # empty locations
        "-":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/empty.png").convert('RGBA'),

        # Flag
        "^":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/flag_top.png").convert('RGBA'),
        "f":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/flag_white.png").convert('RGBA'),
        "I":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/flag_middle.png").convert('RGBA'),

        # starting location
        "M":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/mario.png").convert('RGBA'),

        # Enemies
        "y":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/spiky.png").convert('RGBA'),
        "Y":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/spiky_wings.png").convert('RGBA'),
        "E":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/gomba.png").convert('RGBA'),
        "g":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/gomba.png").convert('RGBA'),
        "G":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/gomba_wings.png").convert('RGBA'),
        "k":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/greenkoopa.png").convert('RGBA'),
        "K":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/greenkoopa_wings.png").convert('RGBA'),
        "r":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/redkoopa.png").convert('RGBA'),
        "R":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/redkoopa_wings.png").convert('RGBA'),
        "X":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/floor.png").convert('RGBA'),
        "#":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/solid.png").convert('RGBA'),

        # Jump through platforms
        "=":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/mushroomtop_middle.png").convert('RGBA'),
        "(":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/mushroomtop_left.png").convert('RGBA'),
        ")":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/mushroomtop_right.png").convert('RGBA'),
        "z":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/mushroomtop.png").convert('RGBA'),
        "|":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/mushroombody.png").convert('RGBA'),

        # Bullet Bill Shooters
        "B":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/bulletbill_head.png").convert('RGBA'),
        "b":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/bulletbill_neck.png").convert('RGBA'),
        "p":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/bulletbill_body.png").convert('RGBA'),

        # Question Mark Blocks
        "?":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/question_powerup.png").convert('RGBA'),
        "@":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/question_powerup.png").convert('RGBA'),
        "Q":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/question_coin.png").convert('RGBA'),
        "!":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/question_coin.png").convert('RGBA'),
        "D":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/question_empty.png").convert('RGBA'),

        # Hidden Blocks
        "1":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/hidden_1up.png").convert('RGBA'),
        "2":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/hidden_coin.png").convert('RGBA'),

        # Brick Blocks
        "S":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/brick.png").convert('RGBA'),
        "C":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/brick_coin.png").convert('RGBA'),
        "U":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/brick_powerup.png").convert('RGBA'),
        "L":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/brick_1up.png").convert('RGBA'),

        # Coin
        "o":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/coin.png").convert('RGBA'),

        # Pipes
        "<":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/tubetop_left.png").convert('RGBA'),
        ">":
            Image.open(
                os.path.dirname(__file__) +
                "/sprites/tubetop_right.png").convert('RGBA'),
        "[":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/tube_left.png").convert('RGBA'),
        "]":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/tube_right.png").convert('RGBA'),
        "O":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/tubetop.png").convert('RGBA'),
        "H":
            Image.open(os.path.dirname(__file__) +
                       "/sprites/tube.png").convert('RGBA'),
    }
    return graphics


def getMarioLevel(levelString, locations=None, graphics=None, tileSize=16):
    if isinstance(levelString, list):
        tempLines = levelString
    else:
        tempLines = levelString.split('\n')
    if graphics is None:
        graphics = getMarioGraphics()

    if isinstance(tempLines[0], list):
        levelLines = levelString
        width = len(levelLines[0])
        height = len(levelLines)
    else:
        levelLines = []
        width = 0
        height = 0
        for line in tempLines:
            line = line.strip()
            if len(line) > 0:
                levelLines.append(line)
                if len(line) > width:
                    width = len(line)
        height = len(levelLines)

    decodedMap = []
    exit_x = -1
    exit_y = -1
    for y in range(height):
        decodedMap.append([])
        for x in range(width):
            char = levelLines[y][x]
            if char == "F":
                exit_x = x
                exit_y = y
                char = "-"
            if char == "%":
                index = 0
                if x < width - 1 and levelLines[y][x + 1] == "%":
                    index += 1
                if x > 0 and levelLines[y][x - 1] == "%":
                    index += 2
                char = ["z", "(", ")", "="][index]
            if char == "*":
                index = 0
                if y > 0 and levelLines[y - 1][x] == "*":
                    index += 1
                if y > 1 and levelLines[y - 2][x] == "*":
                    index += 1
                char = ["B", "b", "p"][index]
            if char in ["T", "t"]:
                singlePipe = True
                topPipe = True
                if (x < width - 1 and levelLines[y][x + 1].lower() == 't') or (
                        x > 0 and levelLines[y][x - 1].lower() == 't'):
                    singlePipe = False
                if y > 0 and levelLines[y - 1][x].lower() == 't':
                    topPipe = False
                if singlePipe:
                    if topPipe:
                        char = "O"
                    else:
                        char = "H"
                else:
                    if topPipe:
                        char = "<"
                        if x > 0 and levelLines[y][x - 1].lower() == 't':
                            char = ">"
                    else:
                        char = "["
                        if x > 0 and levelLines[y][x - 1].lower() == 't':
                            char = "]"
            decodedMap[y].append(char)

    if exit_x > 1:
        decodedMap[1][exit_x] = "^"
        decodedMap[2][exit_x - 1] = "f"
    for y in range(2, exit_y + 1):
        decodedMap[y][exit_x] = "I"

    lvl_image = Image.new("RGBA", (width * tileSize, height * tileSize),
                          (109, 143, 252, 255))
    for y in range(height):
        for tx in range(width):
            x = width - tx - 1
            shift_x = 0
            if decodedMap[y][x] == "f":
                shift_x = 8
            lvl_image.paste(graphics[decodedMap[y][x]],
                            (x * tileSize + shift_x, y * tileSize,
                             (x + 1) * tileSize + shift_x, (y + 1) * tileSize))

    if locations:
        marioWalk = Image.open(
            os.path.dirname(__file__) +
            "/sprites/mario_walk.png").convert('RGBA')
        marioJump = Image.open(
            os.path.dirname(__file__) +
            "/sprites/mario_jump.png").convert('RGBA')
        for loc in locations:
            img = marioWalk
            if 'ground' in loc:
                if not loc['ground']:
                    img = marioJump
            lvl_image.paste(
                img,
                (int(loc['x'] - tileSize / 2), int(loc['y'] - tileSize),
                 int(loc['x'] + tileSize / 2), int(loc['y'])),
                img,
            )
    return lvl_image


def visualize_level(level: Union[np.ndarray, MarioLevel, str],
                    output: Union[str, Path] = None,
                    show_mario=True):
    """Creates an image of the Mario level.

    Args:
        level: The level to plot.
        output: Path to save the image. Set to None to avoid saving.
        show_mario: Whether to include Mario in the level.
    Returns:
        Image: The image of the Mario level.
    """
    if isinstance(level, np.ndarray):
        level = MarioLevel(level).to_str()
    elif isinstance(level, MarioLevel):
        level = level.to_str()

    if show_mario:
        agent = Agent()
        game = MarioGame()
        # Identical settings as in MarioModule.evaluate, except we turn off
        # visualization with False.
        result = game.runGame(agent, level, 20, 0, False)
        locations = [{
            "x": obj.getMarioX(),
            "y": obj.getMarioY(),
            "ground": obj.getMarioOnGround(),
        } for obj in result.getAgentEvents()]
    else:
        locations = []

    lvlImg = getMarioLevel(level, locations)
    if output is not None:
        lvlImg.save(str(output))
    return lvlImg


if __name__ == "__main__":
    visualize_level(
        """\
--------------------------------------------------------
-------------------------------------------------------^
------------------------------------------------------fI
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------------------------------------------I
-------------------X-----------------------------------I
------------------XX-----------------------------------I
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX""",
        "level.png",
    )
    print("Visualized example level in level.png")
