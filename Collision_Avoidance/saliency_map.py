##########################################################################

# DoG saliency [Katramados / Breckon 2011] - reference implementation -

# This implementation:
# Copyright (c) 2020 Ryan Lail, Durham University, UK

##########################################################################

import cv2
import numpy as np

##########################################################################


class SaliencyDoG:

    # Parameters:
    # pyramid_height - n as defined in [Katramados / Breckon 2011]
    # shift - k as defined in [Katramados / Breckon 2011]
    # ch_3 - process colour image on every channel
    # low_pass_filter - toggle low pass filter
    # multi_layer_map - the second version of the algortihm as defined
    #                   in [Katramados / Breckon 2011]

    def __init__(self, pyramid_height=5, shift=5, ch_3=False,
                 low_pass_filter=False, multi_layer_map=False):

        self.pyramid_height = pyramid_height
        self.shift = shift
        self.ch_3 = ch_3
        self.low_pass_filter = low_pass_filter
        self.multi_layer_map = multi_layer_map

        # define storage for pyramid layers only if needed
        if self.multi_layer_map:
            self.u_layers = [None]*self.pyramid_height
            self.d_layers = [None]*self.pyramid_height

    def bottom_up_gaussian_pyramid(self, src):

        # Produce Un - step 1 of algortithm defined in [Katramados
        #                                               / Breckon 2011]
        # Uses a 5 X 5 Gaussian filter

        # u1 = src
        un = src

        if self.multi_layer_map:
            self.u_layers[0] = un

        # perform pyrDown pyramid_height - 1 times, yielding pyramid_height
        # layers
        for layer in range(1, self.pyramid_height):
            un = cv2.pyrDown(un)

            if self.multi_layer_map:
                self.u_layers[layer] = un

        return un

    def top_down_gaussian_pyramid(self, src):

        # Produce D1 - step 2 of algorithm defined in [Katramados
        #                                              / Breckon 2011]

        # d1 = src
        dn = src

        if self.multi_layer_map:
            # place at end of array, to correspond with u_layers
            self.d_layers[self.pyramid_height - 1] = src

        # perform pyrUp pyramid_height - 1 times, yielding pyramid_height
        # layers
        for layer in range(self.pyramid_height-2, -1, -1):
            dn = cv2.pyrUp(dn)

            if self.multi_layer_map:
                self.d_layers[layer] = dn

        return dn

    def saliency_map(self, u1, d1, u1_dimensions):

        # Produce S - step 3 of algorithm defined in [Katramados
        #                                             / Breckon 2011]

        if self.multi_layer_map:

            # Initial MiR Matrix M0
            height, width = u1_dimensions
            mir = np.ones((height, width))

            # Convert pixels to 32-bit floats
            mir = mir.astype(np.float32)

            # Use T-API for hardware acceleration
            mir = cv2.UMat(mir)

            for layer in range(self.pyramid_height):

                # corresponding pyramid layers are in same index pos.
                un = self.u_layers[layer]
                dn = self.d_layers[layer]

                # scale layers to original dimenstions
                un_scaled = cv2.resize(un, (width, height))
                dn_scaled = cv2.resize(dn, (width, height))

                # Calculate Minimum Ratio (MiR) Matrix
                matrix_ratio = cv2.divide(un_scaled, dn_scaled)
                matrix_ratio_inv = cv2.divide(dn_scaled, un_scaled)

                # Caluclate pixelwise min
                pixelwise_min = cv2.min(matrix_ratio, matrix_ratio_inv)
                mir_n = cv2.multiply(pixelwise_min, mir)
                mir = mir_n

        else:

            # Check if u1 & d1 are same size
            # (possible discrepencies from fractional height/width
            # when creating pyramids)

            # resize d1 to u1
            d1 = cv2.resize(d1, (u1_dimensions[1], u1_dimensions[0]))

            # Calculate Minimum Ratio (MiR) Matrix
            matrix_ratio = cv2.divide(u1, d1)
            matrix_ratio_inv = cv2.divide(d1, u1)

            # Caluclate pixelwise min
            mir = cv2.min(matrix_ratio, matrix_ratio_inv)

        # Derive salience by subtracting from scalar 1
        s = cv2.subtract(1.0, mir)

        return s

    def divog_saliency(self, src, src_dimensions):

        # Complete implementation of all 3 parts of algortihm defined in
        # [Katramados / Breckon 2011]

        # Shift image by k^n to avoid division by zero or any number in range
        # 0.0 - 1.0
        src = cv2.add(src, self.shift**self.pyramid_height)

        # Base of Gaussian Pyramid (source frame)
        u1 = src

        un = self.bottom_up_gaussian_pyramid(src)
        d1 = self.top_down_gaussian_pyramid(un)
        s = self.saliency_map(u1, d1, src_dimensions)

        # Normalize to 0 - 255 int range
        s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # low-pass filter as defined by original author
        if self.low_pass_filter:
            avg = cv2.mean(s)
            s = cv2.subtract(s, avg)

        return s

    def generate_saliency(self, src):

        # Convert pixels to 32-bit floats
        src = src.astype(np.float32)

        src_dimensions = src.shape[:2]

        # Use T-API for hardware acceleration
        src = cv2.UMat(src)

        if self.ch_3:

            # Split colour image into RBG channels
            channel_array = cv2.split(src)

            # Generate Saliency Map for each channel
            for channel in range(3):

                channel_array[channel] = self.divog_saliency(
                        channel_array[channel], src_dimensions)

            # Merge back into one grayscale image
            merged_channels = cv2.merge(channel_array)
            gray_merged_channels = cv2.cvtColor(merged_channels,
                                                cv2.COLOR_BGR2GRAY)

            return gray_merged_channels

        else:

            # Convert to grayscale
            src_bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            # Generate Saliency Map
            return self.divog_saliency(src_bw, src_dimensions)

