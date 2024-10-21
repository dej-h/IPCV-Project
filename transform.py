import numpy as np
from matplotlib import pyplot as plt

def solve_homography(u, v):
    N = u.shape[0]
    H = None
    if v.shape[0] != N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None

    # TODO: 1.forming A
    u = np.append(u, np.ones((u.shape[0], 1)), axis=1)
    v = np.append(v, np.ones((v.shape[0], 1)), axis=1)

    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2 * i, 3:6] = -1 * u[i, :]
        A[2 * i, 6:] = v[i, 1] * u[i, :]
        A[2 * i + 1, 0:3] = -1 * u[i, :]
        A[2 * i + 1, 6:] = v[i, 0] * u[i, :]

    U, S, V = np.linalg.svd(A)
    H = V[-1, :]
    H = H.reshape((3, 3))
    # TODO: 2.solve H with A
    return H

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape

    src_point = np.array([[0, 0, 1], [0, h_src, 1], [w_src, 0, 1]]).T

    des_point = np.dot(H, src_point) / np.dot(H, src_point)[-1, :]

    # TODO: 1.meshgrid the (x,y) coordinate pairs

    if direction == 'b':
        H_inv = np.linalg.inv(H)
        DX, DY = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1))

        DX = DX.flatten()
        DY = DY.flatten()
        imagemap = np.stack((DX, DY, np.ones(DX.shape)), axis=0)
        newimagemap = np.dot(H_inv, imagemap) / np.dot(H_inv, imagemap)[-1, :]

        newimagemap = newimagemap[:,
                                  np.all(newimagemap[0, :].reshape(1, -1) < w_src, axis=0) & np.all(
                                      newimagemap[1, :].reshape(1, -1) < h_src, axis=0)]
        newimagemap = newimagemap[:,
                                  np.all(newimagemap[0, :].reshape(1, -1) >= 0, axis=0) & np.all(
                                      newimagemap[1, :].reshape(1, -1) >= 0, axis=0)]

        modsrc = bilinear_interpolation(src, newimagemap)

        reimagemap = np.dot(H, newimagemap) / np.dot(H, newimagemap)[-1, :]
        reimagemap = np.rint(reimagemap).astype(int)
        reimagemap = np.delete(reimagemap, -1, axis=0)

        dst[reimagemap[1, :], reimagemap[0, :], :] = modsrc

    elif direction == 'f':
        src_Area = np.linalg.norm(
            np.cross(src_point[0:2, 1] - src_point[0:2, 0], src_point[0:2, 2] - src_point[0:2, 0]))

        des_Area = np.linalg.norm(
            np.cross(des_point[0:2, 1] - des_point[0:2, 0], des_point[0:2, 2] - des_point[0:2, 0]))
        if src_Area > 3 * des_Area:
            step_inter = 0.9
        elif src_Area > des_Area:
            step_inter = 0.8
        else:
            step_inter = src_Area / des_Area * 0.7

        SX, SY = np.meshgrid(np.arange(xmin, xmax, step_inter), np.arange(ymin, ymax, step_inter))
        SX = SX.flatten()
        SY = SY.flatten()

        imagemap = np.stack((SX, SY, np.ones(SX.shape)), axis=0)

        modsrc = bilinear_interpolation(src, imagemap)

        newimagemap = np.dot(H, imagemap) / np.dot(H, imagemap)[-1, :]

        newimagemap = np.rint(newimagemap).astype(int)
        newimagemap = np.delete(newimagemap, -1, axis=0)

        dst[newimagemap[1, :], newimagemap[0, :], :] = modsrc

    return dst

def bilinear_interpolation(img, cod):
    x = np.array(cod[0, :])
    y = np.array(cod[1, :])

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    I00 = img[y0, x0, :]
    I01 = img[y1, x0, :]
    I10 = img[y0, x1, :]
    I11 = img[y1, x1, :]
    w00 = (x1 - x) * (y1 - y)
    w01 = (x1 - x) * (y - y0)
    w10 = (x - x0) * (y1 - y)
    w11 = (x - x0) * (y - y0)

    return (I00.T * w00).T + (I01.T * w01).T + (I10.T * w10).T + (I11.T * w11).T


def transform(img, canvas, corners):
    h, w, ch = img.shape
    x = np.array([[0, 0],
                [w, 0],
                [w, h],
                [0, h]
                ])
    H = solve_homography(x, corners)
    
    return  warping(img, canvas, H, 0, h, 0, w, direction='f')
if __name__ == "__main__":
    u = np.array([[2, 3], [3, 4], [5, 5], [1, 1]])
    v = np.array([[-2, 3], [-3, 4], [2, 5], [11, 1]])
    h = solve_homography(u, v)
    u = np.append(u, np.ones((4, 1)), axis=1)
    print(np.dot(h, u.T) / np.dot(h, u.T)[-1])
    