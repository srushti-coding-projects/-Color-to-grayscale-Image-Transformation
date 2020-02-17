"""

Name : Srushti Kokare
Implementaion to convert color images to gray scale images using algorithm via Non-linear Mapping
Reference to the paper: http://rosaec.snu.ac.kr/publish/2009/ID/KiJaDeLe-SIGGRAPH-2009.pdf

"""


import numpy as np
import skimage.color as scolor
import math
import cv2


INPUT_IMAGE = "test_image.png"

# define constants as mentioned in the base paper
alpha = 1.0
R = 3.59   # calculated directly from the paper
kbr = 0.5627018266011115


def preprocess_image(inputimg):
    norm_image = cv2.normalize(inputimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    return norm_image


def diff_pixels(lab_x, lab_y):
    delta_L = lab_x[0] - lab_y[0]
    delta_sqL = math.pow(delta_L, 2)

    delta_a = lab_x[1] - lab_y[1]
    delta_sq_a = math.pow(delta_a, 2)

    delta_b = lab_x[2] - lab_y[2]
    delta_sq_b = math.pow(delta_b, 2)

    diff = np.power(
        np.add(delta_sqL, np.power(np.divide(np.multiply(alpha, np.power(np.add(delta_sq_a, delta_sq_b),
                                                                         0.5)), R), 2)),.5)

    return delta_L, delta_sqL, delta_a, delta_b, diff


def h_k_predictor(l, u, v):

    suv = 13 * ((u - 0.20917) ** 2 + (v - 0.48810) ** 2) ** 0.5

    theta = math.atan((u - 0.20917) / (v - 0.44810))

    qtheta = - 0.01585 - 0.03017 * math.cos(theta) - 0.04556 * math.cos(2 * theta) - 0.02677 * math.cos(
        3 * theta) - 0.00295 * math.cos(4 * theta) + 0.14592 * math.sin(theta) + 0.05084 * math.sin(
        2 * theta) - 0.01900 * math.sin(3 * theta) - 0.00764 * math.sin(4 * theta)

    lhk = l + (-0.1340 * qtheta + 0.0872 * kbr) * suv * l

    return lhk


def sign_predictor(delta_L, delta_sqL, delta_a, delta_b, diff, delta_lhk):
    if delta_lhk == 0:

        if delta_L == 0:
            sign = np.sign(np.power(delta_L, 1.5) + np.power(delta_a, 1.5) + np.power(delta_b, 1.5))

        else:
            sign = np.sign(delta_L)
    else:
        sign = np.sign(delta_lhk)

    return diff * sign


def optimize(width, height, diff_Gx, diff_Gy, U, V, Lx, Ly, hue_chroma,lightness, chroma, Ms, bs, target_image):
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            p = diff_Gx[i, j] - Lx[i, j]
            q = diff_Gy[i, j] - Ly[i, j]
            u = np.reshape(U[i, j, :], (9, -1))
            v = np.reshape(V[i, j, :], (9, -1))

            bs = bs + (p * u) + (q * v)
            Ms = Ms + (np.multiply(u, (np.reshape(u, (-1, 1)))) + np.multiply(v, (np.reshape(v, (-1, 1)))))

    X = (Ms + width * height * np.identity(9))
    X = np.linalg.lstsq(X, bs, rcond=-1)[0]
    newX = np.zeros((9, 1))
    newX = np.reshape(X, (9, 1))
    for i in range(width):
        for j in range(height):
            f = hue_chroma[i, j, :].reshape(1, 9, order='F')
            f = f * (newX)
            target_image[i, j] = lightness[i, j] + np.multiply(f, chroma[i, j])[0][0]

    return target_image


def main():

    inputimg = cv2.imread(INPUT_IMAGE, flags=cv2.IMREAD_UNCHANGED)

    if not inputimg.data:
        print("Image Not Found.")
    # exit(1)

    # show input image-
    cv2.imshow('InputImage', inputimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    inputimg = cv2.cvtColor(inputimg, cv2.COLOR_RGB2BGR)

    # print details of image like rows, columns and channel
    print("Dimensions :", inputimg.shape)
    width, height, channels = inputimg.shape
    # print image size
    print("Size : ", inputimg.size)
    # print image type
    print("Type : ", inputimg.dtype)

    # create target image that is black in color

    target_image = np.zeros((width, height, channels))

    pro_image = preprocess_image(inputimg)

    # conversion into color spaces

    Lab = scolor.rgb2lab(pro_image)  # - RGB to LAB
    Luv = scolor.rgb2luv(pro_image)  # - RGB to LUV
    Lch = scolor.lab2lch(pro_image)  # - LAB to LCH

    # display dimensions in color space
    print("CIE L*a*b", Lab.shape)
    print("CIE L*u*v", Luv.shape)
    print("CIE L*c*h", Lch.shape)

    lightness = Lch[:, :, 0]
    chroma = Lch[:, :, 1]
    hue = Lch[:, :, 2]

    hue_chroma = np.zeros((width, height, 9))

    for w in range(width):
        for h in range(height):

            hue_chroma[w, h, :] = np.multiply(chroma[w, h], [math.cos(hue[w, h]),
                                                             math.cos(2 * hue[w, h]),
                                                             math.cos(3 * hue[w, h]),
                                                             math.cos(4 * hue[w, h]),

                                                             math.sin(hue[w, h]),
                                                             math.sin(2 * hue[w, h]),
                                                             math.sin(3 * hue[w, h]),
                                                             math.sin(4 * hue[w, h]),
                                                             1])

    print(hue_chroma.shape)

    diff_Gx = np.zeros((width, height))
    diff_Gy = np.zeros((width, height))

    for w in range(1, width - 1):
        for h in range(1, height - 1):
            delta_Lx, delta_sqLx, delta_ax, delta_bx, dx = diff_pixels(Lab[w + 1, h], Lab[w - 1, h])
            lhk_1x = h_k_predictor(Luv[w + 1, h][0], Luv[w + 1, h][1], Luv[w + 1, h][2])
            lnk_1y = h_k_predictor(Luv[w - 1, h][0], Luv[w - 1, h][1], Luv[w - 1, h][2])

            diff_Gx[w, h] = sign_predictor(delta_Lx, delta_sqLx, delta_ax, delta_bx, dx, lhk_1x - lnk_1y)

            delta_Ly, delta_sqLy, delta_ay, delta_by, dy = diff_pixels(Lab[w, h + 1], Lab[w, h - 1])
            lhk_2x = h_k_predictor(Luv[w, h + 1][0], Luv[w, h + 1][1], Luv[w, h + 1][2])
            lnk_2y = h_k_predictor(Luv[w, h - 1][0], Luv[w, h - 1][1], Luv[w, h - 1][2])

            diff_Gy[w, h] = sign_predictor(delta_Ly, delta_sqLy, delta_ay, delta_by, dy, lhk_2x - lnk_2y)

    # Gradients
    U, V, Z = np.gradient(hue_chroma)
    Lx, Ly = np.gradient(lightness)


    Ms = np.zeros((9, 9))  # 9x9 matrix
    bs = np.zeros((9, 1))  # 9x1 vector

    target_image = optimize(width, height, diff_Gx, diff_Gy, U, V, Lx, Ly, hue_chroma, lightness, chroma, Ms, bs, target_image)

    # show output image
    cv2.imshow('OutputImage', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()