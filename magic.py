import os
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from math import floor

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)

plt.rcParams['figure.dpi'] = 150

def init_args():
    "Initializes command line arguments."
    parser = argparse.ArgumentParser(
                    prog='Autostereogram',
                    description='Creates an autostereogram from a depthmap and a pattern.',
                    epilog='Enjoy the program! :)')

    parser.add_argument('-p', '--pattern', type=str, default='assets/pattern.png')
    parser.add_argument('-d', '--depthmap', type=str, default='assets/depthmap.png')
    parser.add_argument('-s', '--shift', type=float, default=0.2)
    parser.add_argument('-c', '--columns', type=int, default=6)
    parser.add_argument('-i', '--invert', action='store_true')
    parser.add_argument('-o', '--output', type=str, default='result.png')
    return parser.parse_args()


def display(img, colorbar=False):
    "Displays an image."
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 2:
        logging.info("Image is grayscale.")
        i = plt.imshow(img, cmap='gray')
    else:
        logging.info("Image is color.")
        # show color image
        i = plt.imshow(img)

    if colorbar:
        logging.info("Showing colorbar...")
        plt.colorbar(i, shrink=0.5, label='depth')

    logging.info("Showing tight layout...")
    plt.tight_layout()

def rescale(img, factor:float):
    "Rescales an image by a factor."
    rows = img.shape[0]
    cols = img.shape[1]
    new_rows = int(rows * factor)
    new_cols = int(cols * factor)
    return np.array([[img[int(r/factor), int(c/factor)] for c in range(new_cols)] for r in range(new_rows)])


def normalize(depthmap):
    "Normalizes values of depthmap to [0, 1] range."
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap

def histogram(img):
    counts, bins = np.histogram(img)
    plt.hist(bins[:-1], bins, weights=counts)

def autostereogram(depthmap, pattern, shift_amplitude=0.1, invert=False):
    "Creates an autostereogram from depthmap and pattern."
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1.0 - depthmap

    pattern_width = pattern.shape[1]
    autostereogram = np.zeros((*depthmap.shape, pattern.shape[2]), dtype=pattern.dtype)
    # row
    for r in np.arange(autostereogram.shape[0]):
        # column
        for c in np.arange(autostereogram.shape[1]):
            # autostereogram[r, c] = pattern[r % pattern.shape[0], c % pattern_width]
            # if column is within pattern, copy pattern
            if c < pattern_width:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            # else, shift pattern
            else:
                # shift is proportional to depthmap value
                shift = int(depthmap[r, c] * shift_amplitude * pattern_width)
                # copy shifted pattern from the previous column
                autostereogram[r, c] = autostereogram[r, c - pattern_width + shift]

    return autostereogram

def load_file(filename):
    "Loads a depthmap from a file."
    return plt.imread(filename)

def save_file(filename, img):
    "Saves an image to a file."
    plt.imsave(filename, img)

try:
    args = init_args()

    depthmap = load_file(args.depthmap)[:, :, 0]
    depthmap_width = depthmap.shape[1]
    # display(depthmap, colorbar=True)

    pattern = load_file(args.pattern)
    pattern_width = pattern.shape[1]

    columns = args.columns
    scale = floor(pattern_width/columns)/pattern_width
    pattern = rescale(pattern, scale)
    # display(pattern)

    img = autostereogram(depthmap, pattern, args.shift, args.invert)
    display(img)

    save_file(args.output, img)

except Exception as e:
    logging.error(e)

plt.show()
