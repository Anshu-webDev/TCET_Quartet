import cv2 as cv
import numpy as np
from object_detection import ObjectDetection
import imutils
import time
# import easyocr
import os
import detect_2
from pytes import recognize_plate


states = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA',
          'GJ', 'HR', 'HP', 'JH', 'KA', 'KL',
          'MP', 'MH', 'MN', 'ML', 'MZ', 'NL',
          'OD', 'PB', 'RJ', 'SK', 'TN', 'TS',
          'TR', 'UP', 'UK', 'WB', 'AN', 'CH',
          'DD', 'DL', 'JK', 'LA', 'LD', 'PY']


def resultplate(plate):
    result = ""
    j = 0
    for character in plate:
        if character == "[":
            result += "U"
        if character.isalnum():
            result += character
        if character.isdigit():
            j += 1
        else:
            j = 0
        if j == 4:
            break
    if j != 4:
        print('Couldn\'t extract number')
    else:
        while result[0:2] not in states and result != "":
            result = result[2:]
        if result == "":
            print('Couldn\'t Recognize Plate. Try with a different plate')
        else:
            return result


# Initialize object detector
od = ObjectDetection()

video = cv.VideoCapture("D:/Anshu/manthan/dataset/videoh.mp4")
frame_list = []
line_position_fifteen = 800

# Define the codec and create VideoWriter object
# filename = "videoo_crop5.avi"
frames_per_sec = 17.0
my_res = "1080p"


def change_res(video, width, height):
    video.set(3, width)
    video.set(3, height)


STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(video, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current capture device
    # to the resulting resolution
    change_res(video, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


temp_blank = ""
# out = cv.VideoWriter(filename, get_video_type(filename), 25, get_dims(video, my_res))


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


# reader = easyocr.Reader(['en'])
count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    width, height, channel = frame.shape
    # print(width, height, channel)
    count += 1
    # if count %4 != 0:
    #     continue

    # Detect objects on frame

    gaussian_blur = cv.GaussianBlur(frame, (5, 5), 0)
    # canny = cv.Canny(gaussian_blur, 30, 200)

    (class_ids, scores, boxes) = od.detect(gaussian_blur)

    for box in boxes:
        (x, y, w, h) = box
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cords = (x, y, x+h, y+h)
        plate = recognize_plate(frame, cords)
        print(plate)
        # print(x, y, w, h)
        # cropped_image = frame[y:y + h, x:x + w]
        # if y > line_position_fifteen and (y + h) > line_position_fifteen:
        # cv.imshow("crop", cropped_image)
        # resize = cv.resize(cropped_image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        # grayscale = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
        # gaussian_blur = cv.GaussianBlur(grayscale, (5, 5), 0)
        # kernel = np.ones((1, 1), np.uint8)
        # dst = cv.filter2D(gaussian_blur, -1, kernel)
        # threshold, thresh = cv.threshold(dst, 150, 255, cv.THRESH_BINARY)
        # filtered = cv.adaptiveThreshold(dst.astype(np.uint8), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9,
        #                                 41)
        # opening = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)
        # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        # img = image_smoothening(img)
        # or_image = cv2.bitwise_or(img, closing)
        # sharpened_image = unsharp_mask(grayscale)
        # canny = cv.Canny(sharpened_image, 30, 200)
        # cv.imshow('my-sharpened-image.jpg', sharpened_image)
        # cv.waitKey(0)
        # gaussian_blur_license_plate = cv.GaussianBlur(sharpened_image, (5, 5), 0)
        # threshold, thresh = cv.threshold(gaussian_blur_license_plate, 150, 255, cv.THRESH_BINARY)

        # count +=1

        # result = reader.readtext(sharpened_image)

        # if len(result) != 0:
        #     plate = result[0][-2]
        #     # plate = resultplate(plate)
        #     plate = plate.replace("[", "U")
        #     plate = plate.replace(" ", "")
        #     ft = plate[:2]
        #     if ft in states:
        #         if plate != temp_blank:
        #             temp_blank = plate
        #             print(plate)
        #         cv.putText(frame, plate, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    # out.write(frame)
    resized = cv.resize(frame, (1000, 700), interpolation=cv.INTER_AREA)
    cv.imshow("Resize", resized)

    # cv.imshow("Frame", frame)

    # cv.waitKey(0)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# out.release()
video.release()
cv.destroyAllWindows()
