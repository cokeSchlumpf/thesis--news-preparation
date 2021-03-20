import tensorflow as tf
import numpy as np
import warnings

from typing import List, Optional
from tqdm import tqdm

EMPTY_TOKEN = '<EMPTY>'
OOV_TOKEN = '<OOV>'
OOI_TOKEN = '<OOI>'


class Vocabulary:
    """
    Util class to work with vocabularies. E.g. mapping samples to indices and vice versa.
    """

    def __init__(self, name: str, token2index: dict, index2token: dict, default_sample_length: int):
        self.name = name
        self.size = len(token2index)
        self.token2index = token2index
        self.index2token = index2token
        self.default_sample_length = default_sample_length

    @staticmethod
    def builder(name: str = 'default') -> 'VocabularyBuilder':
        return VocabularyBuilder(name)

    def token_to_index(self, token: str) -> int:
        """
        Maps a single token to and index.

        :param token: The token to be mapped.
        :return:
        """
        if token in self.token2index:
            return self.token2index[token]
        else:
            return self.token2index[OOV_TOKEN]

    def index_to_token(self, index: int) -> str:
        """
        Maps a single index into the vocabulary token.

        :param index: The index of the token.
        :return: The actual token
        """

        if index in self.index2token:
            return self.index2token[index]
        else:
            return OOI_TOKEN

    def binary_indices_to_sample(self, indices: np.array) -> str:
        """
        Maps a 2-dimensional one-hot encoded sample to the sample string.

        :param indices: The encoded sample.
        :return: The decoded sample.
        """
        result = []

        for position in indices:
            token_index_where = np.where(position == 1)[0]

            if len(token_index_where) > 0:
                result += [self.index_to_token(token_index_where[0])]
            else:
                result += [OOI_TOKEN]

        result = filter(lambda t: t != EMPTY_TOKEN, result)
        return ' '.join(result)

    def binary_sparse_to_sample(self, indices: tf.SparseTensor) -> str:
        """
        Maps a 2-dimensional one-hot encoded sparse tensor to the sample string.

        :param indices: The encoded sample.
        :return: The decoded sample.
        """
        result = []

        for position in indices.indices:
            result += [self.index_to_token(position[1].numpy())]

        result = filter(lambda t: t != EMPTY_TOKEN, result)
        return ' '.join(result)

    def binary_sparse_to_samples(self, indices: tf.SparseTensor) -> List[str]:
        """
        Maps a 3-dimensional encoded sparse matrix of samples to the list of samples.

        :param indices: The encoded samples.
        :return: The decoded list of samples.
        """

        result = {}

        for indices in tqdm(indices.indices):
            indices = indices.numpy().tolist()
            if indices[0] not in result:
                result[indices[0]] = []

            result[indices[0]] = result[indices[0]] + [self.index_to_token(indices[2])]

        result = [filter(lambda t: t != EMPTY_TOKEN, result[k]) for k in result.keys()]
        result = [' '.join(r) for r in result]
        return result

    def binary_indices_to_samples(self, indices: np.array) -> List[str]:
        """
        Maps a 3-dimensional array encoded list of samples to the list of samples.

        :param indices: The encoded samples.
        :return: The decoded list of samples.
        """

        result = []
        for sample in indices:
            result += [self.binary_indices_to_sample(sample)]

        return result

    def indices_to_sample(self, indices: np.array) -> str:
        """
        Maps a 1-dimensional array of indices to a string.

        :param indices: The encoded sample.
        :return: The decoded sample string.
        """
        result = []

        for idx in indices:
            result += [self.index_to_token(idx)]

        result = filter(lambda t: t != EMPTY_TOKEN, result)

        return ' '.join(result)

    def indices_to_samples(self, indices: np.array) -> List[str]:
        """
        Maps a 2-dimensional array of indices to a set of sample strings.

        :param indices:
        :return:
        """

        result = []
        for sample in indices:
            result += [self.indices_to_sample(sample)]

        return result

    def sample_to_binary_indices(self, sample: str,  max_len: Optional[int] = None) -> np.array:
        """
        Maps a sample to a 2-dimensional array (one-hot encoded); shape: [max_len, voc_size].

        :param sample: The sample to transform.
        :param max_len: The length of the output array.
            If None, the default from the vocabulary will be used.
            If 0, the length will be according to the sample.
        :return: A Numpy array with the transformed sample.
        """

        tokens, max_len = self._calculate_max_len(sample, max_len)

        result = np.zeros([max_len, self.size])
        result[:, self.token_to_index(EMPTY_TOKEN)] = 1.0

        for token_position in range(0, min(max_len, len(tokens))):
            token_index = self.token_to_index(tokens[token_position])
            result[token_position, :] = 0.0
            result[token_position, token_index] = 1.0

        return result

    def sample_to_binary_sparse(self, sample: str, max_len: Optional[int] = None) -> tf.SparseTensor:
        """
        Maps a sample to a 2-dimensional array (one-hot encoded); shape: [max_len, voc_size].

        :param sample: The sample to transform.
        :param max_len: The length of the output array.
            If None, the default from the vocabulary will be used.
            If 0, the length will be according to the sample.
        :return: A Numpy array with the transformed sample.
        """

        tokens, max_len = self._calculate_max_len(sample, max_len)
        indices = []
        values = []

        for token_position in range(0, min(max_len, len(tokens))):
            token_index = self.token_to_index(tokens[token_position])
            indices += [[token_position, token_index]]
            values += [1.0]

        return tf.SparseTensor(indices=indices, values=values, dense_shape=[max_len, self.size])

    def sample_to_indices(self, sample: str, max_len: Optional[int] = None) -> np.array:
        """
        Maps a sample to an array of token indices.

        :param sample: The sample to transform.
        :param max_len: The length of the output array.
            If None, the default from the vocabulary will be used.
            If 0, the length will be according to the sample.
        :return: A Numpy array with the transformed sample.
        """

        tokens, max_len = self._calculate_max_len(sample, max_len)

        result = np.zeros(max_len, dtype='int32')
        result[:] = self.token_to_index(EMPTY_TOKEN)

        for i in range(0, min(max_len, len(tokens))):
            result[i] = self.token_to_index(tokens[i])

        return result

    def samples_to_binary_indices(self, samples, max_len: Optional[int] = None) -> np.array:
        """
        Maps a list of samples to a 3-dimensional array.
        1st dimension will be samples, 2nd indices for token positions, 3rd for token indices (one-hot encoded).
        The shape of the output is [samples.size, max_len, voc_size].

        :param samples: The list/ array of samples.
        :param max_len: If None, the default of the vocabulary will be used.
        :return: The transformed samples.
        """

        if max_len is None:
            max_len = self.default_sample_length

        result = np.zeros([len(samples), max_len, self.size])
        for idx_sample in range(0, len(samples)):
            sample = samples[idx_sample]
            result[idx_sample] = self.sample_to_binary_indices(sample, max_len)

        return result

    def samples_to_binary_sparse(self, samples, max_len: Optional[int] = None) -> tf.SparseTensor:
        """
        Maps a list of samples to a 3-dimensional sparse tensor.
        1st dimension will be samples, 2nd indices for token positions, 3rd for token indices (one-hot encoded).
        The shape of the output is [samples.size, max_len, voc_size].

        :param samples: The list/ array of samples.
        :param max_len: If None, the default of the vocabulary will be used.
        :return: The transformed samples.
        """
        if max_len is None:
            max_len = self.default_sample_length

        indices = []
        values = []

        for idx_sample in tqdm(range(0, len(samples))):
            sample = samples[idx_sample]
            sample_sparse = self.sample_to_binary_sparse(sample, max_len)

            indices += [[idx_sample] + si.numpy().tolist() for si in sample_sparse.indices]
            values += sample_sparse.values.numpy().tolist()

        return tf.SparseTensor(indices=indices, values=values, dense_shape=[len(samples), max_len, self.size])

    def samples_to_indices(self, samples, max_len: Optional[int] = None) -> np.array:
        """
        Maps a list of samples to a 2-dimensional array. 1st dimension will be samples, 2nd indices for tokens.
        The shape of the output is [samples.size, max_len].

        :param samples: The list/ array of samples.
        :param max_len: The size of the 2nd dimension. If None, the default of the vocabulary will be used.
        :return: The transformed samples.
        """

        if max_len is None:
            max_len = self.default_sample_length

        result = np.zeros([len(samples), max_len], dtype='int32')
        for idx_sample in range(0, len(samples)):
            sample = samples[idx_sample]
            result[idx_sample] = self.sample_to_indices(sample, max_len)

        return result

    def _calculate_max_len(self, sample: str, max_len: Optional[int]):
        tokens = sample.split(' ')

        if max_len is None:
            max_len = self.default_sample_length

        if max_len == 0:
            max_len = len(tokens)

        if max_len < len(tokens):
            warnings.warn(f"sample length ({len(tokens)}) is longer than length of output array ({max_len})")

        return tokens, max_len

    def __repr__(self):
        return f"Vocabulary(name='{self.name}', size={self.size}, default_sample_length={self.default_sample_length})"


class VocabularyBuilder:
    """
    Util class to build a vocabulary.
    """

    def __init__(self, name):
        self.name = name
        self.max_sample_len = 0
        self.size = 2
        self._token2index = {
            EMPTY_TOKEN: 0,
            OOV_TOKEN: 1
        }
        self._index2token = {
            0: EMPTY_TOKEN,
            1: OOV_TOKEN
        }
        self._token_count = {}

        pass

    def add_token(self, token) -> 'VocabularyBuilder':
        """
        Adds a new single token to the vocabulary.

        :param token: The token to be added.
        :return: The builder instance
        """
        if token not in self._token2index:
            idx = self.size
            self._token2index[token] = idx
            self._token_count[token] = 1
            self._index2token[idx] = token
            self.size += 1
        else:
            self._token_count[token] += 1

        return self

    def add_sample(self, sample) -> 'VocabularyBuilder':
        """
        Adds a new sample to the vocabulary.

        :param sample: The sample to be added to the vocabulary.
        :return: The builder instance
        """

        tokens = sample.split(' ')
        self.max_sample_len = max(len(tokens), self.max_sample_len)

        for token in tokens:
            self.add_token(token)

        return self

    def build(self, min_token_occurrences: int = 0):
        """
        Creates the Vocabulary instance.

        :param min_token_occurrences: The number of minimum occurrences in the samples
                                      to include the token in the vocabulary.
        :return: The vocabulary instance.
        """

        if min_token_occurrences > 0:
            for token in self._token2index.keys():
                if token is not OOV_TOKEN and token is not EMPTY_TOKEN and \
                        self._token_count[token] < min_token_occurrences:

                    idx = self._token2index[token]
                    del self._index2token[idx]
                    del self._token2index[token]

        return Vocabulary(self.name, self._token2index, self._index2token, self.max_sample_len)
