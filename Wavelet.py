from PIL import Image
import numpy as np
import numba


@numba.njit(fastmath=True)
def DOG(x: float) -> float:

    return 1 / np.exp((x ** 2) / 2) - 0.5 * (1 / np.exp((x ** 2) / 8))


@numba.njit(fastmath=True)
def DOG1(x: float) -> float:
    return 0.125 * x * (1 / np.exp((x ** 2) / 8)) - x * (1 / np.exp((x ** 2) / 2))


@numba.njit(fastmath=True)
def disDOG(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * DOG((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True)
def disDOG1(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * DOG1((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True, parallel=True)
def dxDOG(pixels):
    Xq, Yq = pixels.shape
    Xdec = int(round((np.log(Xq) / np.log(2)) - 1))
    DWT = np.zeros((Yq - 1, Xdec, Xq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Xdec, Xq - 1))
        for m in numba.prange(Xdec):
            for n in numba.prange(Xq - 1):
                for x in numba.prange(Xq - 1):
                    DWT1[m][n] += disDOG(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = np.copy(DWT1)
    IDWT = np.zeros((Xq-1, Yq - 1))
    for y in numba.prange(Yq - 1):
        for x in numba.prange(Xq - 1):
            for i in numba.prange(Xdec):
                for j in numba.prange(Xq - 1):
                    IDWT[x][y] += disDOG1(x, 2 ** (i - 1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 2


@numba.njit(fastmath=True, parallel=True)
def dyDOG(pixels):
    Xq, Yq = pixels.shape
    Ydec = int(round((np.log(Yq) / np.log(2)) - 1))
    DWT = np.zeros((Xq - 1, Ydec, Yq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Ydec, Yq - 1))
        for m in numba.prange(Ydec):
            for n in numba.prange(Yq - 1):
                for x in numba.prange(Yq - 1):
                    DWT1[m][n] += disDOG(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = DWT1.copy()
    IDWT = np.zeros((Yq-1, Xq - 1))
    for x in numba.prange(Xq - 1):
        for y in numba.prange(Yq - 1):
            for i in numba.prange(Ydec):
                for j in numba.prange(Yq - 1):
                    IDWT[x][y] += disDOG1(x, 2**(i-1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 2


def grad(difX, difY, image):
    res = image.copy()
    for x in range(difX.shape[0]):
        for y in range(difY.shape[0]):
            a = int(round(np.sqrt(difX[x][y] ** 2 + difY[x][y] ** 2)))
            res.putpixel((x, y), (a, a, a))
    return res


@numba.njit(fastmath=True)
def WAVE(x: float) -> float:
    return -x * (1 / np.exp((x ** 2) / 2))


@numba.njit(fastmath=True)
def WAVE1(x: float) -> float:
    return (x ** 2) * (1 / np.exp((x ** 2) / 2)) - (1 / np.exp((x ** 2) / 2))


@numba.njit(fastmath=True)
def disWAVE(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * WAVE((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True)
def disWAVE1(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * WAVE1((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True, parallel=True)
def dxWAVE(pixels):
    Xq, Yq = pixels.shape
    Xdec = int(round((np.log(Xq) / np.log(2)) - 1))
    DWT = np.zeros((Yq - 1, Xdec, Xq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Xdec, Xq - 1))
        for m in numba.prange(Xdec):
            for n in numba.prange(Xq - 1):
                for x in numba.prange(Xq - 1):
                    DWT1[m][n] += disWAVE(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = np.copy(DWT1)
    IDWT = np.zeros((Xq-1, Yq - 1))
    for y in numba.prange(Yq - 1):
        for x in numba.prange(Xq - 1):
            for i in numba.prange(Xdec):
                for j in numba.prange(Xq - 1):
                    IDWT[x][y] += disWAVE1(x, 2 ** (i - 1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 6


@numba.njit(fastmath=True, parallel=True)
def dyWAVE(pixels):
    Xq, Yq = pixels.shape
    Ydec = int(round((np.log(Yq) / np.log(2)) - 1))
    DWT = np.zeros((Xq - 1, Ydec, Yq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Ydec, Yq - 1))
        for m in numba.prange(Ydec):
            for n in numba.prange(Yq - 1):
                for x in numba.prange(Yq - 1):
                    DWT1[m][n] += disWAVE(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = DWT1.copy()
    IDWT = np.zeros((Yq-1, Xq - 1))
    for x in numba.prange(Xq - 1):
        for y in numba.prange(Yq - 1):
            for i in numba.prange(Ydec):
                for j in numba.prange(Yq - 1):
                    IDWT[x][y] += disWAVE1(x, 2**(i-1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 6


@numba.njit(fastmath=True)
def MHAT(x: float) -> float:
    return ((2 * (1 / (np.pi ** (1 / 4))))/(np.sqrt(3))) * (1 - x ** 2) * (1 / (np.exp((x ** 2) / 2)))


@numba.njit(fastmath=True)
def MHAT1(x: float) -> float:
    return (2 * np.sqrt(3) * x * (1 / np.exp((x ** 2) / 2)) * ((x ** 2) - 3)) / (3 * np.pi ** (1 / 4))


@numba.njit(fastmath=True)
def disMHAT(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * MHAT((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True)
def disMHAT1(x: float, m: float, n: float) -> float:
    return 1 / (3 ** (m / 2)) * MHAT1((1 / (3 ** m)) * x - n)


@numba.njit(fastmath=True, parallel=True)
def dxMHAT(pixels):
    Xq, Yq = pixels.shape
    Xdec = int(round((np.log(Xq) / np.log(2)) - 1))
    DWT = np.zeros((Yq - 1, Xdec, Xq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Xdec, Xq - 1))
        for m in numba.prange(Xdec):
            for n in numba.prange(Xq - 1):
                for x in numba.prange(Xq - 1):
                    DWT1[m][n] += disMHAT(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = np.copy(DWT1)
    IDWT = np.zeros((Xq-1, Yq - 1))
    for y in numba.prange(Yq - 1):
        for x in numba.prange(Xq - 1):
            for i in numba.prange(Xdec):
                for j in numba.prange(Xq - 1):
                    IDWT[x][y] += disMHAT1(x, 2 ** (i - 1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 6


@numba.njit(fastmath=True, parallel=True)
def dyMHAT(pixels):
    Xq, Yq = pixels.shape
    Ydec = int(round((np.log(Yq) / np.log(2)) - 1))
    DWT = np.zeros((Xq - 1, Ydec, Yq - 1))
    for y in numba.prange(Yq - 1):
        DWT1 = np.zeros((Ydec, Yq - 1))
        for m in numba.prange(Ydec):
            for n in numba.prange(Yq - 1):
                for x in numba.prange(Yq - 1):
                    DWT1[m][n] += disMHAT(x, 2**(m-1), n) * pixels[x][y]
        DWT[y] = DWT1.copy()
    IDWT = np.zeros((Yq-1, Xq - 1))
    for x in numba.prange(Xq - 1):
        for y in numba.prange(Yq - 1):
            for i in numba.prange(Ydec):
                for j in numba.prange(Yq - 1):
                    IDWT[x][y] += disMHAT1(x, 2**(i-1), j) * DWT[y][i][j]
            IDWT[x][y] = abs(IDWT[x][y])
    return IDWT / 6


def normFactor(image):
    result = image.copy()
    pixels = image.load()
    minC, maxC = pixels[0, 0][0], pixels[0, 0][0]
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if minC > pixels[x, y][0]:
                minC = pixels[x, y][0]
            if maxC < pixels[x, y][0]:
                maxC = pixels[x, y][0]
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            a = int(((image.getpixel((x, y))[0] - minC) * 254) / (maxC - minC))
            result.putpixel((x, y), (a, a, a))
    return result