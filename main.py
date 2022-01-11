import pyautogui
import numpy
import cv2
# ca sa mearga cv2 in terminal: pip install opencv-python
import PIL
# ca sa mearga PIL in terminal: pip install pillow
import time


def capture_board(dimensions_of_board):
    image_of_board = numpy.array(PIL.ImageGrab.grab(bbox=dimensions_of_board))
    image_of_board = cv2.cvtColor(image_of_board, cv2.COLOR_BGR2RGB)
    return image_of_board


def image_processing():
    dimensions_of_board = (27, 155, 385, 515)
    original_board = capture_board(dimensions_of_board)

    # cv2.imshow('original_board', original_board)
    # cv2.waitKey(0)
    # 20,21 -> preia tabla de joc cu bulinele de pe ea

    size_of_board = vertical_line_detector(original_board)
    size_of_square = 375 / size_of_board

    print("size of board" , size_of_board)

    board_of_pixels = create_pixel_board(dimensions_of_board, size_of_board, size_of_square)
    board_of_colours = create_colour_board(board_of_pixels, size_of_board)


def create_pixel_board(dimensions_of_board, size_of_board, size_of_square):
    x_dim, y_dim = dimensions_of_board[0:2]
    board_of_pixels = numpy.zeros((size_of_board, size_of_board), dtype=object)
    for i in range(size_of_board):
        for j in range(size_of_board):
            board_of_pixels[i, j] = (int(x_dim + j * size_of_square + 0.5 * size_of_square),
                                     int(y_dim + i * size_of_square + 0.5 * size_of_square))
    return board_of_pixels


def create_colour_board(board_of_pixels, size_of_board):
    dict_of_colours = {(0, 0, 255): 'b', (255, 0, 0): 'r', (0, 128, 0): 'g', (238, 238, 0): 'o', (255, 0, 255): 'p',
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
                board_of_colours[i, j] = '0'
    print(board_of_colours)


def vertical_line_detector(image):
    line_tolerance = 5
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3)
    lines = cv2.HoughLinesP(image=edge_image, rho=1, theta=numpy.pi, threshold=100,
                            lines=numpy.array([]), minLineLength=300, maxLineGap=5)
    lines_to_keep = []
    for line1 in lines:
        for line2 in lines:
            if line1[0][0] < line2[0][0] < line1[0][0] + line_tolerance:
                lines_to_keep.append(line1)
    lines_to_keep = numpy.array(lines_to_keep)
    return len(lines_to_keep) - 1


if __name__ == '__main__':
    image_processing()
