import cv2

IMG_SIZE = 299

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4
}

def preprocess_image(path, method):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=INTERPOLATIONS[method])
    img = img / 255.0
    return img