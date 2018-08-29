import numpy as np
from os.path import dirname
from collections import Counter
from copy import deepcopy
from math import log
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageColor


class Motif:
    """
    The Motif class is a convenience class to compute and plot a position-weight matrix (PWM).
    The only functionality is the plot function. The PWM and corresponding entropy values
    can be accessed using the self.pwm and self.entropies members, if so desired. All uppercase
    alphanumeric characters and the following additional characters can be part of the
    alphabet: "()[]{}<>,.|*".
    """

    def __init__(self, alphabet, sequences = None, pwm = None):
        """ Initialize a motif by providing sequences or a PWM.

        Either a list of sequences or a PWM with shape (sequence length, alphabet length) 
        must be provided.

        Parameters
        ----------
        alphabet : str
            The alphabet of the sequences.
        
        sequences : [str]
            A list of strings. All strings must have the same length.
        
        pwm : numpy.ndarray
            A matrix of shape (sequence length, alphabet length) containing probabilities.
        """
        self.alphabet = alphabet
        if sequences != None:
            self._compute_counts(sequences)
        else:
            self.pwm = deepcopy(pwm)
        self._add_pseudocounts()
        self._compute_entropies()


    def _compute_counts(self, sequences):
        by_pos = zip(*sequences)
        counts_per_pos = map(Counter, by_pos)
        self.pwm = np.empty(len(self.alphabet) * len(sequences[0]))
        self.pwm.shape = (len(sequences[0]), len(self.alphabet))
        for i, pos in enumerate(counts_per_pos):
            for j, char in enumerate(self.alphabet):
                self.pwm[i, j] = pos[char]


    def _add_pseudocounts(self):
        for i, pos in enumerate(self.pwm):
            fun = np.vectorize(lambda x: 0.999*(x/sum(pos)) + 0.001*(1./len(self.alphabet)))
            self.pwm[i] = fun(pos)


    def _compute_entropies(self):
        fun = np.vectorize(lambda x: x*log(x, 2))
        self.entropies = np.empty(self.pwm.shape[0])
        for i, pos in enumerate(self.pwm):
            self.entropies[i] = -sum(fun(pos))


    def plot(self, colors={}, scale=1):
        """ Plot the motif.

        The color of individual letters can be defined via the colors dict using RGB values, e.g.
        {'A': '#FF0000', 'C': '#0000FF'} will result in red A's and blue C's. Non-defined characters
        will be plotted black.

        The alphabets 'ACGT', 'ACGU', and 'HIMS' have predefined colors (that can be overwritten):
        
        '"ACGT" -> {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}
        '"ACGU" -> {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'U': '#CC0000'}
        '"HIMS" -> {'H': '#CC0000', 'I': '#FFB300', 'M': '#00CC00', 'S': '#CC00FF'}

        Using, for instance, a scale parameter of 0.5 halves both height and width of the plot.

        Parameters
        ----------
        colors : dict of char->str
            A dict with individual alphabet characters as keys and hexadecimal RGB specifiers as values.
        
        scale : float
            Adjust the size of the plot (should be > 0).
        
        Returns
        -------
        image : PIL.image.image
            A Pillow image object.
        """

        # prepare colors
        self.colors = deepcopy(colors)
        if self.colors == {}:
            if self.alphabet == 'ACGT':
                self.colors = {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'T': '#CC0000'}
            elif self.alphabet == 'ACGU':
                self.colors = {'A': '#00CC00', 'C': '#0000CC', 'G': '#FFB300', 'U': '#CC0000'}
            elif self.alphabet == 'HIMS':
                self.colors = {'H': '#CC0000', 'I': '#FFB300', 'M': '#00CC00', 'S': '#CC00FF'}
        # translate hex to decimal
        for char in self.colors:
            if len(self.colors[char]) != 7 or not self.colors[char].startswith("#"):
                raise RuntimeError("Error: '{}' is not a valid color specifier.".format(self.colors[char]))
            self.colors[char] = ImageColor.getrgb(self.colors[char])
        # cache all alphabet character images
        img_chars = self._load_characters()
        # prepapre image dimensions
        w_char, h_char = img_chars[self.alphabet[0]].size
        w_col, h_col = w_char, h_char*3
        h_top, h_bottom = 40, 60
        w_total, h_total = w_col + w_col*len(self.pwm) + 40, h_top + h_col + h_bottom
        img_motif = Image.new("RGB", (w_total, h_total), "#ffffff")
        img_draw = ImageDraw.Draw(img_motif)
        # plot axes
        self._add_y_axis(img_motif, img_draw, w_col, h_col, h_top)
        self._add_x_axis(img_motif, img_draw, w_col, h_col, h_top)
        # plot sequence motif
        self._add_motif(img_motif, w_col, h_col, h_top, img_chars)
        # default height is 754 pixels
        if scale != 1:
            w_scaled, h_scaled = int(w_total*scale), int(h_total*scale)
            img_motif = img_motif.resize((w_scaled, h_scaled), Image.BICUBIC)
        for x in img_chars:
            img_chars[x].close()
        return img_motif


    def _load_characters(self):
        folder = dirname(__file__)
        img_chars = {}
        for char in self.alphabet:
            # these three characters can not be used in filenames
            if char == '|':
                img_chars[char] = Image.open("{}/resources/motif/char_except0.png".format(folder))
            elif char == '<':
                img_chars[char] = Image.open("{}/resources/motif/char_except1.png".format(folder))
            elif char == '>':
                img_chars[char] = Image.open("{}/resources/motif/char_except2.png".format(folder))
            elif char == '*':
                img_chars[char] = Image.open("{}/resources/motif/char_except3.png".format(folder))
            else:
                img_chars[char] = Image.open("{}/resources/motif/char{}.png".format(folder, char))
            # change the color if needed
            if char in self.colors:
                #img_chars[char] = img_chars[char].convert('RGBA')
                data = np.array(img_chars[char])
                red, green, blue, _alpha = data.T
                not_white = (red != 255) & (blue != 255) & (green != 255)
                data[..., :-1][not_white.T] = self.colors[char]
                img_chars[char] = Image.fromarray(data)
        return img_chars


    def _trim(self, img):
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        return img.crop(diff.getbbox())


    def _get_and_rotate_bits(self):
        folder = dirname(__file__)
        font_bits = ImageFont.truetype("{}/resources/motif/LiberationSans-Regular.ttf".format(folder), 70)
        img_bits = Image.new("RGB", (500, 500), "#ffffff")
        draw_bits = ImageDraw.Draw(img_bits)
        draw_bits.text((250,250), "bits", (0, 0, 0), font = font_bits)
        img_bits = img_bits.rotate(90)
        return self._trim(img_bits)


    def _add_y_axis(self, img_motif, img_draw, w_col, h_col, h_top):
        # draw the rotated "bits" label
        img_bits = self._get_and_rotate_bits()
        w_bits, h_bits = img_bits.size
        img_motif.paste(img_bits, (w_col//2 - int(1.5*w_bits), h_col//2 - h_bits//2 + h_top))
        img_bits.close()
        # draw y axis
        img_draw.line((w_col, 0+h_top, w_col, h_col+h_top), fill = "#000000", width = 5)
        # draw y ticks and labels
        folder = dirname(__file__)
        font = ImageFont.truetype("{}/resources/motif/LiberationSans-Regular.ttf".format(folder), 50)
        info_content = log(len(self.alphabet), 2)
        ticks = np.arange(0.0, info_content + 0.5, 0.5)
        for x in ticks:
            if x > info_content: break
            y_tick = h_top + h_col - h_col*(x/info_content)
            img_draw.line((w_col-20, y_tick, w_col, y_tick), fill = "#000000", width = 5)
            textwidth, textheight = img_draw.textsize(str(x), font)
            img_draw.text((w_col-25-textwidth, y_tick - textheight//2 - 3),
                          str(x), (0, 0, 0), font = font)


    def _add_x_axis(self, img_motif, img_draw, w_col, h_col, h_top):
        x_tick = w_col + 10
        folder = dirname(__file__)
        font = ImageFont.truetype("{}/resources/motif/LiberationSans-Regular.ttf".format(folder), 50)
        for i, _ in enumerate(self.pwm):
            textwidth, _ = img_draw.textsize(str(i+1), font)
            img_draw.text((x_tick + w_col//2 - textwidth//2, h_col + h_top),
                          str(i+1), (0, 0, 0), font = font)
            x_tick += w_col


    def _add_motif(self, img_motif, w_col, h_col, h_top, img_chars):
        x_tick = w_col + 10
        info_content = log(len(self.alphabet), 2)
        for i, pos in enumerate(self.pwm):
            total = h_col - self.entropies[i] * (h_col/info_content)
            size_chars = [(char, int(pos[x]*total)) for x, char in enumerate(self.alphabet)]
            size_chars.sort(key = lambda x: x[1])
            y_tick = h_col + h_top
            for x in size_chars:
                y_tick = y_tick - x[1]
                scaled = img_chars[x[0]].resize((w_col, max(1, x[1])), Image.BICUBIC)
                img_motif.paste(scaled, (x_tick, y_tick), mask = scaled)
            x_tick += w_col