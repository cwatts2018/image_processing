"""
6.1010 Spring '23 Lab 1: Image Processing
"""

#!/usr/bin/env python3

#import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


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


# def set_pixel(image, index, color):
#     image["pixels"][index] = color


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


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
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
    by the "mode" parameter.
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
    # blue_gill = load_greyscale_image("test_images/bluegill.png")
    # blue_gill_inv = inverted(blue_gill)
    # save_greyscale_image(blue_gill_inv, "test_images/blue_gill_inv.png")
    # pigbird = load_greyscale_image("test_images/pigbird.png")
    # kernel = {"width": 13, "height": 13, "values":
    #           [0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           1,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0,
    #           0,0,0,0,0,0,0,0,0,0,0,0,0]}
    # pigbird_cor = correlate(pigbird, kernel, "wrap")
    # print (type(pigbird_cor))
    # save_greyscale_image(pigbird_cor, "test_images/pigbird_wrap.png")
    # cat = load_greyscale_image("test_images/cat.png")
    # cat_blurred = blurred(cat, 13)
    # save_greyscale_image(cat_blurred, "test_images/cat_blurred_wrap.png")
    # python = load_greyscale_image("test_images/python.png")
    # python_sharpened = sharpened(python, 11)
    # save_greyscale_image(python_sharpened, "test_images/python_sharpened.png")
    # python = load_greyscale_image("test_images/python.png")
    # construct = load_greyscale_image("test_images/construct.png")
    # construct_edges = edges(construct)
    # save_greyscale_image(construct_edges, "test_images/construct_edges.png")
    img = load_greyscale_image("test_images/centered_pixel.png")
    construct_edges = edges(img)
    save_greyscale_image(construct_edges, "test_images/center_pixel_edges.png")


