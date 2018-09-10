import numpy as np


class One_Hot_Encoder:
    """
    The One_Hot_Encoder class provides functions to encode a string over a
    given alphabet into an integer matrix of shape (len(string), len(alphabet))
    where each row represents a position in the string and each column
    represents a character from the alphabet. Each row has exactly one 1 at the
    matching alphabet character and consists of 0s otherwise.
    """

    def __init__(self, alphabet):
        """ Initialize the object with an alphabet.
        
        Parameters
        ----------
        alphabet : str
            The alphabet that will be used for encoding/decoding (e.g. "ACGT").
        """
        self.alphabet = alphabet
        self.table = {symbol: i for i, symbol in enumerate(alphabet)}
        self.table_rev = {v: k for k, v in self.table.items()}
    
    def encode(self, sequence):
        """ Encode a sequence into a one-hot integer matrix.
        
        The sequence should only contain characters from the alphabet provided to __init__.

        Parameters
        ----------
        sequence : str
            The sequence that should be encoded.

        Returns
        -------
        one_hot: numpy.ndarray
            A numpy array with shape (len(sequence), len(alphabet)).
        """
        one_hot = np.zeros((len(sequence), len(self.table)), np.uint8)
        one_hot[np.arange(len(sequence)), np.fromiter(map(self.table.__getitem__, sequence), np.uint32)] = 1
        return one_hot

    def decode(self, one_hot):
        """ Decode a one-hot integer matrix into the original sequence.

        Parameters
        ----------
        one_hot : numpy.ndarray
            A one-hot matrix (e.g. as created by the encode function).

        Returns
        -------
        sequence: str
            The sequence that is represented by the one-hot matrix.
        """
        return ''.join(map(self.table_rev.__getitem__, np.argmax(one_hot, axis=1)))