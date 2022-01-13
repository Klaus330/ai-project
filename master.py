import copy
import pydirectinput as pydirectinput
import pyautogui
import numpy
import cv2
import PIL
import time

class Cell:
    def __init__(self, position, color='B', visited=False, goal=False):
        self.position = position
        self.color = color
        self.isInAPath = False
        self.isGoal = goal
        self.visited = visited
        self.previous = self

    def isEmpty(self):
        return self.color == 'B'

    def isBorder(self):
        return self.color == 'X'

    def isPartOfPath(self):
        return self.isInAPath

    def isSameColor(self, color):
        return self.color == color

def capture_board(dimensions_of_board):
    image_of_board = numpy.array(PIL.ImageGrab.grab(bbox=dimensions_of_board))
    image_of_board = cv2.cvtColor(image_of_board, cv2.COLOR_BGR2RGB)
    return image_of_board

def image_processing():
    dimensions_of_board = (63, 181, 499, 619)
    square_length = 363

    original_board = capture_board(dimensions_of_board)

    # cv2.imshow('original_board', original_board)
    # cv2.waitKey(0)

    size_of_board = vertical_line_detector(original_board)
    size_of_square = square_length / size_of_board
    print(size_of_square)
    print("size of board", size_of_board)

    board_of_pixels = create_pixel_board(dimensions_of_board, size_of_board, size_of_square)
    board_of_colours = create_colour_board(board_of_pixels, size_of_board)
    return (board_of_colours, size_of_board, board_of_pixels)

def create_pixel_board(dimensions_of_board, size_of_board, size_of_square):
    x_dim, y_dim = dimensions_of_board[0:2]
    board_of_pixels = numpy.zeros((size_of_board, size_of_board), dtype=object)
    for i in range(size_of_board):
        for j in range(size_of_board):
            board_of_pixels[i, j] = (int(x_dim + j * size_of_square + 0.5 * size_of_square),
                                     int(y_dim + i * size_of_square + 0.5 * size_of_square))
    return board_of_pixels

def create_colour_board(board_of_pixels, size_of_board):
    dict_of_colours = {(0, 0, 255): 'a', (255, 0, 0): 'r', (0, 128, 0): 'g', (255, 127, 0): 'o', (238, 238, 0): 'y',
                       (255, 0, 255): 'p',
                       (128, 0, 128): 'z', (0, 255, 255): 'c', (0, 128, 128): 't', (0, 0, 139): 'd',
                       (166, 166, 166): 'q', (189, 183, 107): 's', (0, 255, 0): 'l', (165, 42, 42): 'm',
                       (255, 255, 255): 'w'}
    board_of_colours = numpy.zeros((size_of_board, size_of_board), dtype=str)
    for i in range(size_of_board):
        for j in range(size_of_board):
            if pyautogui.pixel(board_of_pixels[i, j][0], board_of_pixels[i, j][1]) in dict_of_colours.keys():
                board_of_colours[i, j] = dict_of_colours[
                    pyautogui.pixel(board_of_pixels[i, j][0], board_of_pixels[i, j][1])]
            else:
                board_of_colours[i, j] = 'B'
    return board_of_colours

def vertical_line_detector(image):
    line_tolerance = 5
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3)
    # cv2.imshow('original_board', image)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(image=edge_image, rho=1, theta=numpy.pi, threshold=100,
                            lines=numpy.array([]), minLineLength=300, maxLineGap=5)
    lines_to_keep = []
    for line1 in lines:
        for line2 in lines:
            if line1[0][0] < line2[0][0] < line1[0][0] + line_tolerance:
                lines_to_keep.append(line1)
    lines_to_keep = numpy.array(lines_to_keep)

    return len(lines_to_keep) - 1

def generate_board(coloredBoard, n, m):
    print(coloredBoard)
    # exit()
    color_board = []
    for i in range(n + 2):
        row = []
        for j in range(m + 2):
            if i == 0 or j == 0 or i == n + 1 or j == m + 1:
                row.append(Cell((i, j), 'X', True))
            else:
                if coloredBoard[i - 1][j - 1] != "B":
                    queue.append((i, j))
                row.append(Cell((i, j), coloredBoard[i - 1][j - 1].upper(), False,
                                True if coloredBoard[i - 1][j - 1] != "B" else False))
        color_board.append(row)
    display_board(color_board)
    return color_board

def display_board(given_board):
    print()
    for i in range(cols + 2):
        for j in range(rows + 2):
            print(given_board[i][j].color, end=" ")
        print()

    # for i in range(cols + 2):
    #     for j in range(rows + 2):
    #         print(given_board[i][j].isGoal, end=" ")
    #     print()

def validMove(neighbour, current, move):
    value = neighbour.isEmpty() or (current.color == neighbour.color and not neighbour.visited)
    return value

def finalPath1(neighbourCell, currentCell):
    if neighbourCell.color == currentCell.color and neighbourCell.visited == True and neighbourCell != currentCell.previous:
        return True

def finalPath2(neighbourdCell):
    if not neighbourdCell.isEmpty() or neighbourdCell.isBorder():
        return True

def quick_queue(unsolved_board, i1, j1, i2, j2):
    for index in range(0, len(queue)):
        if queue[index][0] == i1 and queue[index][1] == j1:
            unsolved_board[i2][j2].color = unsolved_board[i1][j1].color
            unsolved_board[i1][j1].visited = True
            unsolved_board[i2][j2].visited = True
            unsolved_board[i2][j2].previous = unsolved_board[i1][j1]
            queue[index] = (i2, j2)
    return unsolved_board

def corner_move_method(unsolved_board):
    # Corner Left-Up
    if not unsolved_board[2][1].isEmpty() and unsolved_board[1][1].isEmpty():
        unsolved_board = quick_queue(unsolved_board, 2, 1, 1, 1)
    if not unsolved_board[1][2].isEmpty() and unsolved_board[1][1].isEmpty():
        unsolved_board = quick_queue(unsolved_board, 1, 2, 1, 1)

    # Corner Rigt-Up
    if not unsolved_board[1][cols - 1].isEmpty() and unsolved_board[1][cols].isEmpty():
        unsolved_board = quick_queue(unsolved_board, 1, cols - 1, 1, cols)
    if not unsolved_board[2][cols].isEmpty() and unsolved_board[1][cols].isEmpty():
        quick_queue(unsolved_board, 2, cols, 1, cols)

    # Corner Left-Down
    if not unsolved_board[cols - 1][1].isEmpty() and unsolved_board[cols][1].isEmpty():
        unsolved_board = quick_queue(unsolved_board, cols - 1, 1, cols, 1)
    if not unsolved_board[cols][2].isEmpty() and unsolved_board[cols][1].isEmpty():
        unsolved_board = quick_queue(unsolved_board, cols, 2, cols, 1)

    # Corner Right-Down
    if not unsolved_board[cols - 1][cols].isEmpty() and unsolved_board[cols][cols].isEmpty():
        unsolved_board = quick_queue(unsolved_board, cols - 1, cols, cols, cols)
    if not unsolved_board[cols][cols - 1].isEmpty() and unsolved_board[cols][cols].isEmpty():
        unsolved_board = quick_queue(unsolved_board, cols, cols - 1, cols, cols)

    return unsolved_board

def force_moves_method(unsolved_board):
    global queue
    global queue2
    global cols, rows
    nr = 0
    copy_of_board = copy.deepcopy(unsolved_board)
    while len(queue) > 1:
        if nr == 100:
            if not areBoardsIdentical(unsolved_board, copy_of_board):
                copy_of_board = copy.deepcopy(unsolved_board)
                nr = 0
            else:
                break

        nr += 1
        i, j = queue.pop(0)
        currentCell = unsolved_board[i][j]
        ok = False
        neighbourLeft = unsolved_board[i][j - 1]
        neighbourRight = unsolved_board[i][j + 1]
        neighbourTop = unsolved_board[i - 1][j]
        neighbourBottom = unsolved_board[i + 1][j]

        if (finalPath1(neighbourLeft, currentCell) or finalPath1(neighbourRight, currentCell) or \
                finalPath1(neighbourBottom, currentCell) or finalPath1(neighbourTop, currentCell)):
            continue

        if finalPath2(neighbourLeft) and finalPath2(neighbourBottom) and finalPath2(neighbourRight) and finalPath2(
                neighbourTop):
            continue

        # check right
        if not validMove(neighbourTop, currentCell, "right") and not validMove(neighbourLeft, currentCell,
                                                                               "right") and not \
                validMove(neighbourBottom, currentCell, "right") and validMove(neighbourRight, currentCell,
                                                                               "right"):
            neighbourRight.color = currentCell.color
            neighbourRight.previous = currentCell
            neighbourRight.visited = True
            currentCell.visited = True
            queue.append((i, j + 1))
            continue

        # check left
        elif not validMove(neighbourTop, currentCell, "left") and validMove(neighbourLeft, currentCell,
                                                                            "left") and not \
                validMove(neighbourBottom, currentCell, "left") and not validMove(neighbourRight, currentCell,
                                                                                  "left"):
            neighbourLeft.color = currentCell.color
            neighbourLeft.visited = True
            neighbourLeft.previous = currentCell
            currentCell.visited = True
            queue.append((i, j - 1))
            continue

        # check top
        elif validMove(neighbourTop, currentCell, "top") and not validMove(neighbourLeft, currentCell,
                                                                           "top") and not \
                validMove(neighbourBottom, currentCell, "top") and not validMove(neighbourRight, currentCell,
                                                                                 "top"):
            neighbourTop.color = currentCell.color
            neighbourTop.visited = True
            neighbourTop.previous = currentCell
            currentCell.visited = True
            queue.append((i - 1, j))
            continue

        # check bottom
        elif not validMove(neighbourTop, currentCell, "bottom") and not validMove(neighbourLeft, currentCell,
                                                                                  "bottom") and \
                validMove(neighbourBottom, currentCell, "bottom") and not validMove(neighbourRight, currentCell,
                                                                                    "bottom"):
            neighbourBottom.color = currentCell.color
            neighbourBottom.visited = True
            neighbourBottom.previous = currentCell
            currentCell.visited = True
            queue.append((i + 1, j))
            continue

        queue.append((i, j))

    return unsolved_board

def reverse_color_matrix(unsolved_board):
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if isinstance(unsolved_board[i][j].color, int):
                unsolved_board[i][j].color = 'B'
    return unsolved_board

def fill(unsolved_board, coords, val):
    global dx, dy
    queue3 = []
    queue3.append(coords)
    while (len(queue3) > 0):
        i, j = queue3.pop()
        for vecin in range(0, 4):
            neighboardCell = unsolved_board[i + dx[vecin]][j + dy[vecin]]
            if (neighboardCell.color == 'B'):
                queue3.append((i + dx[vecin], j + dy[vecin]))
                unsolved_board[i + dx[vecin]][j + dy[vecin]].color = val

    return unsolved_board

def group_method(unsolved_board):
    global queue
    val = 1
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if unsolved_board[i][j].color == 'B':
                unsolved_board[i][j].color = val
                unsolved_board = fill(unsolved_board, (i, j), val)
                display_board(unsolved_board)
                val += 1
    print(queue)
    dictionary = {}
    for cell in queue:
        i, j = cell
        neighboardCell = []
        for vecin in range(0, 4):
            neighboardCell.append(unsolved_board[i + dx[vecin]][j + dy[vecin]])

        for insula in range(1, val + 1):
            for vecin in neighboardCell:
                if vecin.color == insula:
                    if str(unsolved_board[i][j].color) in '0987654321':
                        continue

                    if unsolved_board[i][j].color not in dictionary.keys():
                        dictionary[unsolved_board[i][j].color] = []

                    dictionary[unsolved_board[i][j].color].append(insula)
                    break
    print(dictionary)
    color = ""
    for insula in range(1, val):
        nr = 0
        rez = ''
        for keys in dictionary.keys():
            count = 0
            for val in dictionary[keys]:
                if val == insula:
                    count += 1
            if count == 2:
                if nr == 1:
                    rez = 'impossible'
                    break
                nr += 1
                color = keys
        if rez != 'impossible':
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    if unsolved_board[i][j].color == insula:
                        unsolved_board[i][j].color = color
                        unsolved_board[i][j].visited = True

    unsolved_board = reverse_color_matrix(unsolved_board)

    return unsolved_board

def areBoardsIdentical(board1, board2):
    same = True
    for i in range(1, cols + 1):
        for j in range(1, rows + 1):
            if board1[i][j].color != board2[i][j].color:
                same = False
                break
    return same

def solve(unsolved_board):
    original_unsolved_board = copy.deepcopy(unsolved_board)

    unsolved_board = force_moves_method(unsolved_board)
    if not areBoardsIdentical(unsolved_board, original_unsolved_board):
        unsolved_board = solve(unsolved_board)

    unsolved_board = group_method(unsolved_board)
    if not areBoardsIdentical(unsolved_board, original_unsolved_board):
        unsolved_board = solve(unsolved_board)

    return unsolved_board

def generate_solved_board(game_board):
    array = []
    for i in range(1, cols + 1):
        for j in range(1, rows + 1):
            array.append(game_board[i][j].color)

    return array

def move_mouse(game_board):
    solved_board3 = generate_solved_board(game_board)
    print(solved_board3)
    list_of_colors = list(set(solved_board3))
    instances_of_colors = {i: solved_board3.count(i) for i in solved_board3}

    colors_ranges = {k: list(range(1, v + 1)) for k, v in instances_of_colors.items()}

    dictionary_of_moves = {}
    for color, moves in colors_ranges.items():
        array_of_moves = numpy.zeros((rows, cols), dtype=int)

        # adauga capetele
        for i in range(rows + 1):
            for j in range(cols + 1):
                if board[i + 1][j + 1].isGoal and board[i + 1][j + 1].color == color:
                    if moves[0] == 1:
                        array_of_moves[i][j] = moves.pop(0)
                        solved_board3[j + (i * rows)] = 'B'
                    else:
                        array_of_moves[i][j] = moves.pop(-1)
                        solved_board3[j + (i * rows)] = 'B'

        # adauga restul de mutari
        while len(moves) > 0:
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):

                    if array_of_moves[i - 1][j - 1] == moves[0] - 1 or array_of_moves[i - 1][j - 1] == moves[-1] + 1:

                        currentCell = board[i][j]
                        neighbourLeft = board[i][j - 1]
                        neighbourRight = board[i][j + 1]
                        neighbourTop = board[i - 1][j]
                        neighbourBottom = board[i + 1][j]

                        # right
                        if neighbourRight.isSameColor(color) and not neighbourLeft.isSameColor(
                                color) and not neighbourTop.isSameColor(color) and not neighbourBottom.isSameColor(
                            color):
                            array_of_moves[i - 1][j] = moves.pop(0) if array_of_moves[i - 1][j - 1] == moves[
                                0] - 1 else moves.pop(-1)
                            currentCell.color = "B"

                        # left
                        if not neighbourRight.isSameColor(color) and neighbourLeft.isSameColor(
                                color) and not neighbourTop.isSameColor(color) and not neighbourBottom.isSameColor(
                            color):
                            array_of_moves[i - 1][j - 2] = moves.pop(0) if array_of_moves[i - 1][j - 1] == moves[
                                0] - 1 else moves.pop(-1)
                            currentCell.color = "B"

                        # top
                        if not neighbourRight.isSameColor(color) and not neighbourLeft.isSameColor(
                                color) and neighbourTop.isSameColor(color) and not neighbourBottom.isSameColor(color):
                            array_of_moves[i - 2][j - 1] = moves.pop(0) if array_of_moves[i - 1][j - 1] == moves[
                                0] - 1 else moves.pop(-1)
                            currentCell.color = "B"

                        # bottom
                        if not neighbourRight.isSameColor(color) and not neighbourLeft.isSameColor(
                                color) and not neighbourTop.isSameColor(color) and neighbourBottom.isSameColor(color):
                            array_of_moves[i][j - 1] = moves.pop(0) if array_of_moves[i - 1][j - 1] == moves[
                                0] - 1 else moves.pop(-1)
                            currentCell.color = "B"

                    if len(moves) == 0:
                        break
                if len(moves) == 0:
                    break

        dictionary_of_moves[color] = array_of_moves
    return dictionary_of_moves

def draw_solution(moves_dict, pixelboard):
    for color, moves_array in moves_dict.items():
        moveNr = 1
        old_x, old_y = 0, 0
        while True:
            position = numpy.where(moves_array == moveNr)

            if len(position[0]) == 0:
                break

            i, j = position[0][0], position[1][0]
            x, y = pixelboard[i][j]
            if moveNr == 1:
                pydirectinput.moveTo(x, y, duration=0.01)
                pydirectinput.click(duration=0.01)
            else:
                pydirectinput.moveTo(x, y, duration=0.01)

            if moveNr == moves_array.max():
                pydirectinput.click(duration=0.01)

            moveNr += 1
            time.sleep(0.3)

dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]
queue = []
queue2 = []
rows = cols = 5
board = None


def main():
    global rows, cols, board

    colorBoard, boardSize, board_of_pixels = image_processing()
    rows = cols = boardSize
    print(cols)
    board = generate_board(colorBoard, rows, cols)

    board = corner_move_method(board)
    solved_board = solve(board)
    display_board(board)
    moves = move_mouse(solved_board)
    draw_solution(moves, board_of_pixels)
    time.sleep(1)
    # main()


main()
