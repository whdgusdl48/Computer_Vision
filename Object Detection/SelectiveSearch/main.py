# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectiveSearch

def main():

    # loading astronaut image
    # img = cv2.imread('0204x4.png')
    img = skimage.data.astronaut()
    # perform selective search
    img_lbl, regions = selectiveSearch.selective_search(
        img, scale=500, sigma=0.9, min_size=50)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 0
    for x, y, w, h in candidates:
        print(x, y, w, h)
        # if i == 7:
        #     break
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        i += 1

    plt.show()
if __name__ == "__main__":
    main()