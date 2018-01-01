class Detector:

    START_Y = 390
    START_WINDOW_SIZE = 24
    OVERLAP = 0.5
    PIX_PER_CELL = 8


    def __init__(self, classifier):
        self.classifier = classifier

    def draw_bounds(self, img, output_img=None):
        if not output_img:
            output_img = img

        # extract hog features from entire search region (START_Y to height, all X)

        # sliding window to find all possible matches
        for window in self._generate_windows(img):
            print(window)

    def _generate_windows(self, img):
        width, height = img.shape[1], img.shape[0]

        start_y = self.START_Y
        end_y = height
