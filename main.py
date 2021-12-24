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

    size_of_board = vertical_line_detector(original_board)


def vertical_line_detector(image):
    line_tolerance = 5
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3)
    lines = cv2.HoughLinesP(image=edge_image, rho=1, theta=numpy.pi, threshold=100,
                            lines=numpy.array([]), minLineLength=400, maxLineGap=5)
    lines_to_keep = []
    for line1 in lines:
        for line2 in lines:
            if line1[0][0] < line2[0][0] < line1[0][0] + line_tolerance:
                lines_to_keep.append(line1)
    lines_to_keep = numpy.array(lines_to_keep)
    print(lines_to_keep)


if __name__ == '__main__':
    image_processing()
