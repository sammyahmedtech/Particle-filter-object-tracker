import numpy as np
import math
import random
import cv2
from PIL import Image
import os.path as path
import os


class ParticleFilter:

    def __init__(self, dirpath, M, R, Q, lambda_psi,
                 draw_rectangles = True, draw_all_particles = True,
                 uWeight = 1, ground_truth_path = None):

        self.M = M
        self.R = R
        self.Q = Q
        self.lambda_psi = lambda_psi

        self.draw_rectangles = draw_rectangles
        self.draw_all_particles = draw_all_particles

        self.dataset = self.readDataset(dirpath)
        self.imgx = self.dataset[0].shape[1]  # Original image size.
        self.imgy = self.dataset[0].shape[0]
        self.xdim = self.imgx  # Size for scaled images.
        self.ydim = self.imgy
        self.scale = 1

        self.xMeanOld = 0
        self.yMeanOld = 0

        self.uWeight = uWeight

        self.score = 0

        if ground_truth_path is not None:
            self.display_ground_truth = True
            self.ground_truth_path = ground_truth_path
        else:
            self.display_ground_truth = True
            self.ground_truth_path = ''


    def scale_images(self, scale):
        """Sets the new scaled image dimension values. """
        self.scale = scale
        self.xdim = int(self.imgx * scale)
        self.ydim = int(self.imgy * scale)


    def readDataset(self, dirpath):
        """Reads the dataset and returns the image matrices in an array. """
        img_list = []
        for name in os.listdir(dirpath):
            if name.endswith('.jpg'):
                img_list.append(name)

        img_list.sort()

        if (path.isdir(dirpath)):
            x = np.array([np.array(Image.open(path.join(dirpath, filename))) for filename in img_list])
            return x
        else:
            return "Error no directory found"


    def preProc(self, img):
        """Performs the OpenCV people detection on a single image. Returns the
        middle point of the detected rectangles (can be multiple), and the image
        with the rectangle drawn on it. """
        # init detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # detect people
        (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)

        # draw bounding boxes
        if np.size(rects) > 0:
            # row position, column new rect
            pos = np.zeros((2, rects.shape[0]))

            for i, (x, y, w, h) in enumerate(rects):
                if self.draw_rectangles:
                    cv2.rectangle(
                        img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                pos[0, i] = float(x + w / 2) / self.xdim
                pos[1, i] = float(y + h / 2) / self.ydim
        else:
            return img, np.array([])

        return img, pos


    def draw_particles(self, S, img, clr = (0, 255, 0)):
        """Draws all the particles. """
        for i in range(S.shape[1]):
            cv2.circle(
                img, (int(S[0, i]*self.xdim), int(S[1, i]*self.ydim)), 2, clr, 2)

        return img


    def draw_particle_mean(self, S, img, clr = (0, 0, 255)):
        """Draws a circle at the mean position of the particles. """
        xc, yc = self.particle_mean(S)
        cv2.circle(
            img, (int(xc*self.xdim), int(yc*self.ydim)), 3, clr, 3)

        return img


    def initParticleSet(self, imgWidth, imgHeight):
        """Initializes the particles randomly uniformly over the area. """
        x = np.random.uniform(0, imgWidth, self.M)
        y = np.random.uniform(0, imgHeight, self.M)
        w = np.ones(self.M) * (1. / self.M)

        S = np.vstack((x, y, w))

        return S


    def particle_mean(self, S):
        """Returns the mean x and y positions of the particles. """
        return np.mean(S[0, :]), np.mean(S[1, :])


    def getU(self, S, iter):
        """Returns a vector that contains how the particle mean moved in the
        previous two steps. If the particle filter is in the first two
        iterations it returns zero. """

        if (iter == 3):
            xMean, yMean = self.particle_mean(S)
            self.xMeanOld = xMean
            self.yMeanOld = yMean
        elif iter > 3:
            xMean, yMean = self.particle_mean(S)
            deltaXMean = xMean - self.xMeanOld
            deltaYMean = yMean - self.yMeanOld

            self.yMeanOld = yMean
            self.xMeanOld = xMean

            return self.uWeight * np.array([[deltaXMean], [deltaYMean]])

        return np.array([[0], [0]])


    def predict(self, S, iter):
        """Propagate particles according to estimated motion model. The Motion
        is estimated from previous particle means. Adds diffusion to the
        prediction. """
        S_bar = np.zeros(S.shape)

        nx = S.shape[0] - 1

        # Motion model.
        u = self.getU(S, iter)

        diffusion = np.multiply(np.random.randn(nx, self.M), np.diag(self.R)[np.newaxis, :].T)

        S_bar[0:nx, :] = S[0:nx, :] + u + diffusion
        S_bar[nx, :] = S[nx, :]

        return S_bar


    def systematic_resample(self, S_bar):
        """Perform systematic resampling of the particles. """
        M = S_bar.shape[1]
        nx = S_bar.shape[0] - 1
        S = np.zeros(S_bar.shape)

        density_function = np.cumsum(S_bar[nx, :])  # Cumulative density function.
        r0 = random.random() / M
        r = r0 + (np.arange(1, M + 1) - 1) / float(M)  # r0 + (m - 1)/M

        A1 = np.tile(density_function, (M, 1))  # Repeat CDF vertically.
        A2 = np.tile(r[np.newaxis, :].T, (1, M))  # Repeat r horizontally.

        indices = np.argmax(A1 >= A2, axis=1)  # i = min CDF(j) >= r0 + (m - 1)/M

        S = S_bar[:, indices]  # Resample.

        S[nx, :] = 1 / float(M)  # Reset weights.

        return S

    def weight(self, S_bar, psi, outlier):
        """Weigh the particles according to the probabilities in psi. """
        psi_inliers = psi[np.invert(outlier), :]  # Discard outlier measurements.

        psi_max = psi[np.argmax(np.sum(psi, 1)), :].reshape(
            (1, S_bar.shape[1]))

        psi_inliers = psi_max

        nx = S_bar.shape[0] - 1
        if psi_inliers.size > 0:
            weights = np.prod(psi_inliers, axis=0)  # Multiply probabilities.
            weights = weights / np.sum(weights)  # Normalize weights.
            S_bar[nx, :] = weights

        return S_bar


    def get_observation_probabilities(self, S_bar, z):
        """Calculate probability of each particle given each measurement.
        Returns the probabilities matrix psi and a vector outlier which tells
        if a measurement is an outlier or not. """
        n = z.shape[1]
        nx = S_bar.shape[0] - 1
        dim = nx

        z_pred = np.tile(S_bar[0:dim, :], (1, n))  # [x x ... x]
        z_obs = np.reshape(np.repeat(z, self.M), (dim, n * self.M))  # [z1 ... z1 zn ... zn]

        nu = z_obs - z_pred  # True observation minus predicted observation.

        exp_term = -0.5 * np.sum(
            np.multiply(np.dot(nu.T, np.linalg.inv(self.Q)).T, nu), axis=0)
        psis = 1 / (2 * math.pi * math.sqrt(np.linalg.det(self.Q))) * np.exp(exp_term)

        psi = np.reshape(psis, (n, self.M))  # Rows: measurements, columns: particles.

        outlier = np.mean(psi, axis=1) < self.lambda_psi

        return outlier, psi


    def draw_ground_truth(self, img, corners_list, clr = (0, 255, 0)):
        """Draws a rectangle on the image by connecting the corners in the
        ground truth with lines. """
        thick = 2

        tc = (np.asarray(corners_list).astype(float)*self.scale).astype(int)

        try:
            cv2.line(img, (tc[0], tc[1]), (tc[2], tc[3]), clr, thick)
            cv2.line(img, (tc[2], tc[3]), (tc[4], tc[5]), clr, thick)
            cv2.line(img, (tc[4], tc[5]), (tc[6], tc[7]), clr, thick)
            cv2.line(img, (tc[6], tc[7]), (tc[0], tc[1]), clr, thick)

            x_mean, y_mean = self.get_ground_truth_mean(corners_list)
            cv2.circle(img, (int(x_mean), int(y_mean)), 3, clr, 3)

        except Exception as e:
            print(e)

        return img


    def get_ground_truth_mean(self, corners_list):
        """Returns the mean value of the ground truth corner coordinates. """
        tc = np.asarray(corners_list).astype(float)*self.scale

        try:
            return np.mean(tc[0::2]), np.mean(tc[1::2])
        except Exception as e:
            print(e)
            return 0, 0


    def accuracy(self, xMean, yMean, truth_corners):
        """Increments the variable self.score if the particle mean is inside
        of the ground truth bounding box. """
        x1 = (int(float(truth_corners[0]) * self.scale))
        y1 = (int(float(truth_corners[1]) * self.scale))
        x2 = (int(float(truth_corners[2]) * self.scale))
        y2 = (int(float(truth_corners[3]) * self.scale))
        x4 = (int(float(truth_corners[6]) * self.scale))
        y4 = (int(float(truth_corners[7]) * self.scale))

        A = np.array([x1, y1])
        B = np.array([x2, y2])
        D = np.array([x4, y4])
        M = np.array(([xMean * self.xdim, yMean * self.ydim]))

        if (0 < np.dot(M - A, B - A)) and (np.dot(M - B, B - A) < 0) \
                and (0 < np.dot(M - A, D - A)) and (np.dot(M - D, D - A) < 0):
            self.score += 1
            return 'True'

        return 'False'


    def run_particle_filter(self):
        """Runs the particle filter. Initializes particles, performs the
        predict and update steps for each image in the dataset. """
        if self.display_ground_truth:
            f = open(self.ground_truth_path, 'r')

        S = self.initParticleSet(1, 1)

        total_rects = 0
        total_outliers = 0

        errors = np.array([])

        for i in range(0, self.dataset.shape[0]):
            print(i)

            img = self.dataset[i]
            img = cv2.resize(img, (self.xdim, self.ydim))

            if i > 0:
                S_bar = self.predict(S, i)
                img, observation = self.preProc(img)
                if observation.size > 0:

                    outlier, psi = self.get_observation_probabilities(
                        S_bar, observation)
                    total_rects += outlier.size
                    total_outliers += np.sum(outlier)

                    S_bar = self.weight(S_bar, psi, outlier)

                    S = self.systematic_resample(S_bar)
                else:
                    S = S_bar

                img = self.draw_particles(S, img)
                img = self.draw_particle_mean(S, img)

                if self.display_ground_truth:
                    line = f.readline()
                    truth_coords = line.split(',')
                    img = self.draw_ground_truth(
                        img, truth_coords, (255, 255, 255))

                    xpmean, ypmean = self.particle_mean(S)
                    xtruth_mean, ytruth_mean = self.get_ground_truth_mean(
                    truth_coords)
                    mean_error = math.sqrt(
                        (xpmean*self.xdim - xtruth_mean)**2 +
                        (ypmean*self.ydim - ytruth_mean)**2
                    )

                    errors = np.hstack((errors, mean_error))
                xmean, ymean = self.particle_mean(S)
                self.accuracy(xmean, ymean, truth_coords)

            cv2.imshow('', img)
            k = cv2.waitKey(10)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break;

        # Accuracy
        acc = (float(self.score)/float(self.dataset.shape[0])) * 100
        print('Accuracy: {} %'.format(acc))
        print('Error mean: {}, variance {}'.format(
            np.mean(errors), np.var(errors)))
        print('Total observations: {}, outliers: {}'.format(
            total_rects, total_outliers))

        try:
            f.close()
        except:
            pass


def main():
    path_prefix = '/home/'
    path_end = 'iceskater1'

    M = 100             # Number of particles.
    R_val = 0.02        # Value used for process noise covariance.
    Q_val = 0.01      # Value used for measurement noise covariance.
    lambda_psi = 0.1    # Threshold for outlier detection.
    scale = 0.75        # Downscaling of images to speed up computation.
    uWeight = 0.5       # A weight on the estimated motion model.

    draw_rectangles = False # Draw rectangles for the OpenCV people detect.

    R = np.diag([1., 1.]) * R_val
    Q = np.diag([1., 1.]) * Q_val

    directory_path = path_prefix + path_end
    #directory_path = 'iceskater1'

    ground_truth_path = directory_path + '/groundtruth.txt'

    p = ParticleFilter(directory_path, M, R, Q, lambda_psi,
                       draw_rectangles=draw_rectangles, uWeight=uWeight,
                       ground_truth_path=ground_truth_path)

    p.scale_images(scale)

    p.run_particle_filter()



if __name__ == '__main__':
    main()
