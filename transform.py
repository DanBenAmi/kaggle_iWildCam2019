import cv2


class CLACHE:
    """
    ref: https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    def __call__(self, train_img):
        # Color balancing
        img_lab = cv2.cvtColor(train_img, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(img_lab)
        img_l = self.clahe.apply(l)
        img_clahe = cv2.merge((img_l, a, b))

        return cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)


class SimpleWhiteBalancing:

    def __init__(self, p=0.4):
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(p)

    def __call__(self, train_img):
        return self.wb.balanceWhite(train_img)


class WhiteBalancing:
    def __init__(self, sat_threshold=0.9):
        self.wb = cv2.xphoto.createGrayworldWB()
        self.wb.setSaturationThreshold(sat_threshold)

    def __call__(self, train_img):
        return self.wb.balanceWhite(train_img)


class WhiteBalancing2:
    def __init__(self, sat_threshold=0.99):
        self.wb = cv2.xphoto.createLearningBasedWB()
        self.wb.setSaturationThreshold(sat_threshold)

    def __call__(self, train_img):
        return self.wb.balanceWhite(train_img)
