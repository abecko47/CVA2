from imageio import imread, imsave
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HistogramProvider:
    COLOR_RANGE = 255

    def __init__(self, source, target):
        self.source = imread(source)
        self.target = imread(target)
        self.source_path = source
        self.target_path = target

    def plot(self):
        source_img = cv2.imread(self.source_path)
        fig, axs = plt.subplots(2)
        axs[0].hist(source_img.ravel(), self.COLOR_RANGE - 1)
        axs[1].imshow(source_img)
        axs[0].set_title("Reference image")

        plt.show()

        target = cv2.imread(self.target_path)
        fig, axs = plt.subplots(2)
        axs[0].hist(target.ravel(), self.COLOR_RANGE - 1)
        axs[1].imshow(target)
        axs[0].set_title("Image to be adjusted")

        plt.show()

        result = cv2.imread("result.jpg")
        fig, axs = plt.subplots(2)
        axs[0].hist(result.ravel(), self.COLOR_RANGE - 1)
        axs[1].imshow(result)
        axs[0].set_title("Histogram matched image")

        plt.show()

    def normalize_histogram(self, source_new_axis, target_no_axis):
        img_hist, bins = np.histogram(source_new_axis[:, :, 0].flatten(), self.COLOR_RANGE)
        tint_hist, bins = np.histogram(target_no_axis[:, :, 0].flatten(), self.COLOR_RANGE)

        cdf_src = img_hist.cumsum()
        cdf_src = (255 * cdf_src / cdf_src[-1]).astype(np.uint8)

        cdf_tint = tint_hist.cumsum()
        cdf_tint = (255 * cdf_tint / cdf_tint[-1]).astype(np.uint8)

        return cdf_src, cdf_tint, bins

    def adap_hist_matching(self):
        img_res = self.source[:, :, np.newaxis]
        src_img = self.source[:, :, np.newaxis]
        target_img = self.target[:, :, np.newaxis]

        src_cdf, target_cdf, bins = self.normalize_histogram(src_img, target_img)

        source_interp = np.interp(src_img[:, :, 0].flatten(), bins[:-1], src_cdf, 0)
        target_interp = np.interp(source_interp, target_cdf, bins[:-1])

        img_res[:, :, 0] = target_interp.reshape((src_img.shape[0], src_img.shape[1]))
        self.save_images(img_res, src_img)

    def save_images(self, img_res, src_img):
        try:
            imsave("result.jpg", img_res)
        except:
            imsave("result_reshape.jpg", img_res.reshape((self.source.shape[0], src_img.shape[1])))
