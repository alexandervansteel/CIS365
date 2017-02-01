import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random


myID, game_map = hlt.get_init()
hlt.send_init("TeamIndigo")


def find_nearest_enemy_direction(square):
    direction = NORTH
    max_distance = min(game_map.width, game_map.height) / 2
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while current.owner == myID and distance < max_distance:
            distance += 1
            current = game_map.get_target(current, d)
        if distance < max_distance:
            direction = d
            max_distance = distance
    return direction

def find_high_production(square):
    direction = NORTH
    production = square.production
    max_distance = min(game_map.width, game_map.height) / 2
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while current.production < (square.production * 1.5) and distance < max_distance:
            distance += 1
            current = game_map.get_target(current, d)
            if distance < max_distance:
                direction = d
                max_distance = d
    return direction

def average_production(square):
    direction = NORTH
    square_count = 0
    total_production = square.production
    max_distance = 8
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while distance < max_distance:
            distance += 1
            current = game_map.get_target(current, d)
            total_production += current.production
            square_count += 1
    return total_production / square_count

def heuristic(square):
    if square.owner == 0 and square.strength > 0:
        return square.production / square.strength
    if square.owner == myID and square.strength > 150:
        return 0
    else:
        # return total potential damage caused by overkill when attacking this square
        return sum(neighbor.strength for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))

def get_move(square):
    target, direction = max(((neighbor, direction) for direction, neighbor in enumerate(game_map.neighbors(square))
                                if neighbor.owner != myID),
                                default = (None, None),
                                key = lambda t: heuristic(t[0]))
    if target is not None and target.strength < square.strength:
        return Move(square, direction)
    elif square.strength < square.production * 5:
        return Move(square, STILL)

    border = any(neighbor.owner != myID for neighbor in game_map.neighbors(square))
    if not border:
        if square.production < average_production(square) and square.strength < 20:
            return Move(square, find_high_production(square))
        else:
            return Move(square, find_nearest_enemy_direction(square))
    else:
        #wait until we are strong enough to attack
        return Move(square,STILL)

while True:
    game_map.get_frame()
    moves = [get_move(square) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
