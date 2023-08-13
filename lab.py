"""
6.101 Spring '23 Lab 2: Image Processing 2
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
#import math
from PIL import Image


# VARIOUS FILTERS
def get_red_img(color_image):
    """
    Returns the greyscale image of the red component of a color image.
    """
    height = color_image["height"]
    width = color_image["width"]
    red_img = {"height": height, "width": width, "pixels": []}
    for tup in color_image["pixels"]:
        red_img["pixels"].append(tup[0])
    return red_img

def get_green_img(color_image):
    """
    Returns the greyscale image of the green component of a color image.
    """
    height = color_image["height"]
    width = color_image["width"]
    green_img = {"height": height, "width": width, "pixels": []}
    for tup in color_image["pixels"]:
        green_img["pixels"].append(tup[1])
    return green_img

def get_blue_img(color_image):
    """
    Returns the greyscale image of the blue component of a color image.
    """
    height = color_image["height"]
    width = color_image["width"]
    blue_img = {"height": height, "width": width, "pixels": []}
    for tup in color_image["pixels"]:
        blue_img["pixels"].append(tup[2])
    return blue_img

def combine_rgb_images(red_img, green_img, blue_img):
    """
    Combines and returns the color image given the greyscale images of the
    red, green, and blue components.
    """
    height = red_img["height"]
    width = red_img["width"]
    filt_image = {"height": height, "width": width, "pixels": []}
    for index in range(height*width):
        tup = (red_img["pixels"][index],
               green_img["pixels"][index],
               blue_img["pixels"][index])
        filt_image["pixels"].append(tup)
    return filt_image

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_filter(image):
        """
        Given a colored image represented as a dictionary with keys "height",
        "width", and "pixels", that hold integer, integer, and an array of
        tuples with pixel rgb values respectively. For example:
            {'height': 3, 'width': 2, 'pixels': [(255, 0, 0), (39, 143, 230)]}
        Returns the image with a filter applied as a new iamge.
        """

        #get each rgb component image
        red_img = get_red_img(image)
        green_img = get_green_img(image)
        blue_img = get_blue_img(image)

        #apply greyscale filter to each rgb image
        filt_red_img = filt(red_img)
        filt_green_img = filt(green_img)
        filt_blue_img = filt(blue_img)

        #combine individual filtered rbg images
        filt_image = combine_rgb_images(filt_red_img,
                                        filt_green_img,
                                        filt_blue_img)

        return filt_image

    return color_filter

def make_blur_filter(kernel_size):
    """
    Given the size of the kernel, returns a function that takes a greyscale
    image and blurs it, returning a new image.
    """
    def blur_filter(image):
        return blurred(image, kernel_size)
    return blur_filter

def make_sharpen_filter(kernel_size):
    """
    Given the size of the kernel, returns a function that takes a greyscale
    image and sharpens it, returning a new image.
    """
    def sharpen_filter(image):
        return sharpened(image, kernel_size)
    return sharpen_filter

def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def apply_filters(image):
        filt_image = image

        #for every color filter in filters, apply the filter on image
        for filt in filters:
            filt_image = filt(filt_image)

        return filt_image

    return apply_filters


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    while ncols > 0:
        grey = greyscale_image_from_color_image(image)
        energy = compute_energy(grey)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        image = image_without_seam(image, seam)
        ncols -= 1
    return image

def custom_feature(image):
    """
    Applies an emboss filter to a greyscale image. Returns a new image.
    """
    kernel = {"width": 5, "height": 5, "values": [1, 0, 0,0,0,
                                                0, 1, 0, 0,0,
                                                0, 0, 0, 0,0,
                                                0,0,0,-1,0,
                                                0,0,0,0,-1]}
    embossed_img = correlate(image, kernel, "extend")
    return round_and_clip_image(embossed_img)


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    greyscale_img = {"height": image["height"],
                     "width": image["width"],
                     "pixels": []}
    for tup in image["pixels"]:
        total = round(0.299*tup[0] + 0.587*tup[1] + 0.114*tup[2])
        greyscale_img["pixels"].append(total)
    return greyscale_img


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """

    cum_energy = {"height": energy["height"],
                  "width": energy["width"],
                  "pixels": energy["pixels"][0:energy["width"]]}

    #for every pixel, find the minimm energy path by looking at cumulative
    #energy paths stored in rows above in cum_energy
    for row in range(1, energy["height"]):
        for col in range(energy["width"]):

            min_energy_sum = get_pixel(energy, row, col)

            #find adjacent pixels
            pix_above = get_pixel(cum_energy, row-1, col)
            if col >= 1:
                pix_left = get_pixel(cum_energy, row-1, col-1)
            else:
                pix_left = -1
            if col < energy["width"]-1:
                pix_right = get_pixel(cum_energy, row-1, col+1)
            else:
                pix_right = -1

            #find minimum adjacent pixel
            if pix_left != -1 and pix_right != -1: #both side adjacent pixels exists
                min_adjacent = min(pix_above, pix_left, pix_right)
            elif pix_left != -1: #only left adjacent pixel exists
                min_adjacent = min(pix_above, pix_left)
            elif pix_right != -1: #only right adjaent pixel exists
                min_adjacent = min(pix_above, pix_right)
            else: #only above pixel exists
                min_adjacent = pix_above

            min_energy_sum += min_adjacent

            # put min_energy_sum into row, col
            cum_energy["pixels"].append(min_energy_sum)

    return cum_energy


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """

    height = cem["height"]
    width = cem["width"]

    #find min value pixel in bottom row
    bottom_row = cem["pixels"][width*(height-1):]
    min_pixel = min(bottom_row)
    index = width*(height-1) + bottom_row.index(min_pixel)
    indices = [index]
    col = index%width


    for row in range(height-1, 0, -1):

        #find adjacent pixels
        pix_above = get_pixel(cem, row-1, col)
        if col >= 1:
            pix_left = get_pixel(cem, row-1, col-1)
        else:
            pix_left = -1
        if col < width-1:
            pix_right = get_pixel(cem, row-1, col+1)
        else:
            pix_right = -1

        #find minimum adjacent pixel
        if pix_left != -1 and pix_right != -1: #both side adjacent pixels exists
            min_adjacent = min(pix_left, pix_above, pix_right)
        elif pix_left != -1: #only left adjacent pixel exists
            min_adjacent = min(pix_left, pix_above)
        elif pix_right != -1: #only right adjaent pixel exists
            min_adjacent = min(pix_above, pix_right)
        else: #only above pixel exists
            min_adjacent = pix_above

        #get new columns
        if min_adjacent == pix_left:
            col -= 1
        elif min_adjacent == pix_above:
            pass
        else:
            col += 1
        indices.append(get_pixel_index(cem, row-1, col))

    return indices

def get_pixel_index(image, row, col):
    """
    Given the row and column of a pixel in image, returns the index of that
    pixel.
    """
    width = image["width"]
    return (width*row)+col

def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    height = image["height"]
    width = image["width"]
    length = len(image["pixels"])
    modified_img = {"height": height, "width": width-1, "pixels": []}

    for index in range(length):
        if index not in seam:
            modified_img["pixels"].append(image["pixels"][index])

    return modified_img

#IMAGE PROCESSING 1 GREYSCALE FUNCTIONS
def get_pixel(image, row, col):
    """
    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    Row: the row of the desired pixel (starting from 0)

    Col: the column of the desired pixel (starting from 0)

    Returns
    -------
    The pixel color value in an image's certain row and column

    """
    index = row*image["width"] + col
    return image["pixels"][index]

def apply_per_pixel(image, func):
    """
    Applies a function func to an inputted image

    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    func : A function to apply to each pixel of image

    Returns
    -------
    A new image with func applied to all its pixels

    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }

    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            result["pixels"].append(new_color) #set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    """
    Inverts an image's color values (ie 0 -> 255).

    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    Returns
    -------
    A new image with its colors inverted.

    """
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS
def get_pixel_w_edge(image, row, col, edge_behavior):
    """
    Returns the color value of a certain pixel in an image.

    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    Row: the row of the desired pixel (starting from 0)

    Col: the column of the desired pixel (starting from 0)

    edge_behavior: a string - "zero", "wrap" or "extend" that describes how to
    how to handle out-of-bounds pixels:
        zero - assigns black pixels to the image on its edges
        wrap - wraps the image around the edges
        extend- extends the image's edges

    """
    index = 0
    width = image["width"]
    height = image["height"]
    if row<0 or col<0 or row>=height or col>=width:
        if edge_behavior == "zero":
            return 0
        elif edge_behavior == "extend":
            if row<0:
                if col<0:
                    return image["pixels"][0]
                elif col>=width:
                    return image["pixels"][width-1]
                return image["pixels"][col]
            elif row>=height:
                if col<0:
                    return image["pixels"][(height-1)*width] #bottom left pixel
                elif col>=width:
                    return image["pixels"][height*width-1] #bottom right pixel
                return image["pixels"][(height-1)*width + col]
            elif col<0:
                return image["pixels"][row*width]
            elif col>=width:
                return image["pixels"][(row+1)*width-1]
        elif edge_behavior == "wrap":
            new_row = row
            new_col = col
            if row > 0:
                new_row = (row%height)
            elif row < 0:
                new_row = height - abs(row)%height
            if col > 0:
                new_col = (col%width)
            elif col < 0:
                new_col = width - abs(col)%width
            return get_pixel(image, new_row, new_col)

    index = row*width + col
    return image["pixels"][index]

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Kernel is a dictionary with the keys: height, width, and values, where rows
    and cols have integer values, and values' value is a list of floats
    """
    edge_str = boundary_behavior.lower()

    if edge_str not in ("zero", "extend", "wrap"):
        return None

    correlated = {"width": image["width"],
                          "height": image["height"],
                          "pixels": []}

    for row in range(image["height"]): #for every pixel
        for col in range(image["width"]):

            #collect nearby pixels same size as kernel
            nearby_pixels = {"width": kernel["width"],
                                  "height": kernel["height"],
                                  "values": []}
            extend_side = kernel["width"]//2
            extend_updown = kernel["height"]//2
            for new_row in range(row-extend_updown, row+extend_updown+1):
                for new_col in range(col-extend_side, col+extend_side+1):
                    color = get_pixel_w_edge(image, new_row, new_col, edge_str)
                    nearby_pixels["values"].append(color)
                # start_i = new_row*image["width"] + col - extend_sideways
                # end_i = new_row*image["width"] + col + extend_sideways
                # nearby_pixels["values"].extend(image["pixels"][start_i:end_i+1])

            #find linear combination
            result = 0
            for pixel, multiplier in zip(nearby_pixels["values"], kernel["values"]):
                result += float(pixel)*multiplier

            #add final pixel value to return dictionary
            correlated["pixels"].append(result)

    return correlated

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    pixels = image["pixels"][:]
    for index, pixel in enumerate(pixels):
        if isinstance(pixels[index], float):
            pixels[index] = round(pixel)
        if pixels[index] < 0:
            pixels[index] = 0
        if pixels[index] > 255:
            pixels[index] = 255
    return {"width": image["width"], "height": image["height"], "pixels": pixels}

# FILTERS
def create_kernel(side_length):
    """
    Creates and returns a nxn kernel with values that add u to 1,
    where n is equal to the side_length input. A dictionary representation
    of the kernel is returned with keys: width, height, and values of the
    kernel

    """
    total_vals = side_length**2
    vals = [1/total_vals] * total_vals
    kernel = {"width": side_length, "height": side_length, "values": vals}
    return kernel

def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = create_kernel(kernel_size)
    blurred_image = correlate(image, kernel, "extend")
    return round_and_clip_image(blurred_image)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.

def sharpened(image, n):
    """
    Sharpens an image.

    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    n: The size of an nxn kernel to sharpen image iwth

    Returns
    -------
    A sharpened image represented by a dictionary

    """
    blur_kernel = create_kernel(n)
    blurred_image = correlate(image, blur_kernel, "extend")
    result = {"width": image["width"], "height": image["height"], "pixels": []}
    for pixel, blurred_pixel in zip(image["pixels"], blurred_image["pixels"]):
        result["pixels"].append(2*pixel-blurred_pixel)
    return round_and_clip_image(result)

def edges(image):
    """
    Emphasizes the edges in an image.

    Parameters
    ----------
    image : An image represented by a dictionary with keys: "width", "height",
    and "pixels", where width and height have integer values, and pixels
    is a list of color values from 0 to 255 inclusive.

    Returns
    -------
    An image represented by a dictionary with the edges emphasized

    """
    krow = {"width": 3, "height": 3, "values": [-1, -2, -1,
                                                0, 0, 0,
                                                1, 2, 1]}
    kcol = {"width": 3, "height": 3, "values": [-1, 0, 1,
                                                -2, 0, 2,
                                                -1, 0, 1]}
    corr_row = correlate(image, krow, "extend")
    corr_col = correlate(image, kcol, "extend")
    result = {"width": image["width"], "height": image["height"], "pixels": []}
    for corr_row_pixel, corr_col_pixel in zip(corr_row["pixels"], corr_col["pixels"]):
        result["pixels"].append((corr_row_pixel**2 + corr_col_pixel**2)**(1/2))
    return round_and_clip_image(result)


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    #inv_cat = inverted(load_greyscale_image('test_images/cat.png'))
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # inv_color_cat = color_inverted(load_color_image("test_images/cat.png"))
    # save_color_image(inv_color_cat, "test_images/color_cat.png")
    # blur_filter = make_blur_filter(9)
    # color_blur_filt = color_filter_from_greyscale_filter(blur_filter)
    # inv_color_python = color_blur_filt(load_color_image("test_images/python.png"))
    # save_color_image(inv_color_python, "test_images/blur_color_python.png")
    # sharp_filter = make_sharpen_filter(7)
    # color_sharp_filt = color_filter_from_greyscale_filter(sharp_filter)
    # save_color_image(sharp_color_sparrow, "test_images/sharp_color_sparrow.png")
    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # frog_filt = filt(load_color_image("test_images/frog.png"))
    # save_color_image(frog_filt, "test_images/frog_filt.png")
    two_cats = load_color_image("test_images/mushroom.png")
    custom_filt = color_filter_from_greyscale_filter(custom_feature)
    custom = custom_filt(two_cats)
    save_color_image(custom, "test_images/mushroom_custom.png")
