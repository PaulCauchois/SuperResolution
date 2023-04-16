from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def nearest_neighbor(im, new_shape):
    match len(im.shape):
        case 2:
            h, v = im.shape
            P = np.pad(im, ((1, 1), (1, 1)))
        case 3:
            h, v, _ = im.shape
            P = np.pad(im, ((1, 1), (1, 1), (0, 0)))
        case _:
            raise ValueError("Initial array doesn't represent 2d image.")
    neighbors_x = np.around(np.tile(np.arange(0, new_shape[0]), (new_shape[1], 1)) * h / new_shape[0]).astype(int)
    neighbors_y = np.around(np.tile(np.arange(0, new_shape[1]), (new_shape[0], 1)).T * v / new_shape[1]).astype(int)
    return np.flip(np.rot90(P[neighbors_x, neighbors_y], axes=(1, 0)), axis=1)


def bilinear(im, new_shape, step=lambda x: x):

    match len(im.shape):
        case 2:
            h, v = im.shape
            P = np.pad(im, ((1, 1), (1, 1)))
            neighbors_x = np.tile(np.arange(0, new_shape[0]), (new_shape[1], 1)) * h / new_shape[0]
            neighbors_y = np.tile(np.arange(0, new_shape[1]), (new_shape[0], 1)).T * v / new_shape[1]
            int_x = neighbors_x.astype(int)
            int_y = neighbors_y.astype(int)
            float_x = step((neighbors_x - int_x))
            float_y = step((neighbors_y - int_y))
            return np.flip(np.rot90(P[int_x, int_y] * (1 - float_x) * (1 - float_y) +
                                    P[int_x + 1, int_y] * float_x * (1 - float_y) +
                                    P[int_x, int_y + 1] * (1 - float_x) * float_y +
                                    P[int_x + 1, int_y + 1] * float_x * float_y, axes=(1, 0)), axis=1).astype(np.uint8)
        case 3:
            h, v, _ = im.shape
            P = np.pad(im, ((1, 1), (1, 1), (0, 0)))
            neighbors_x = np.tile(np.arange(0, new_shape[0]), (new_shape[1], 1)) * h / new_shape[0]
            neighbors_y = np.tile(np.arange(0, new_shape[1]), (new_shape[0], 1)).T * v / new_shape[1]
            int_x = neighbors_x.astype(int)
            int_y = neighbors_y.astype(int)
            float_x = step((neighbors_x - int_x)[:, :, np.newaxis])
            float_y = step((neighbors_y - int_y)[:, :, np.newaxis])
            # Broadcast en (:,:,1) sur un nouvel axe, pour pouvoir faire le calcul avec plusieurs channels.
            return np.flip(np.rot90(P[int_x, int_y] * (1 - float_x) * (1 - float_y) +
                                    P[int_x + 1, int_y] * float_x * (1 - float_y) +
                                    P[int_x, int_y + 1] * (1 - float_x) * float_y +
                                    P[int_x + 1, int_y + 1] * float_x * float_y, axes=(1, 0)), axis=1).astype(np.uint8)
        case _:
            raise ValueError("Initial array doesn't represent 2d image.")


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    im = np.array(Image.open("imgx2.png"))
    plt.imshow(im)
    plt.show()
    nshape = (2*im.shape[0], 2*im.shape[1])

    b_im = bilinear(im, nshape)
    bnew = Image.fromarray(b_im)
    bnew.save("bilinear.png")

    s_im = bilinear(im, nshape, step=lambda x: 3 * x ** 2 - 2 * x ** 3)
    snew = Image.fromarray(b_im)
    snew.save("smoothstep.png")

    nn_im = nearest_neighbor(im, nshape)
    nnew = Image.fromarray(nn_im)
    nnew.save("NN.png")

    diff = Image.fromarray(np.abs(b_im.astype(int) - s_im.astype(int)).astype(np.uint8))
    diff.save("diff.png")
