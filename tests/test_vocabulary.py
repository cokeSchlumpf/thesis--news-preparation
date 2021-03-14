from lib.vocabulary import Vocabulary

SAMPLES = [
    'the quick brown fox jumps over the lazy dog.',
    'lorem ipsum dolor sit amet, consectetuer adipiscing elit.',
    'quick brown rabbit'
]


class TestVocabulary:
    def test_vocabulary(self):
        voc = Vocabulary\
            .builder('sample')\
            .add_sample(SAMPLES[0])\
            .add_sample(SAMPLES[1])\
            .build()

        """
        print(voc.token2index)
        print(voc.sample_to_indices(SAMPLES[2]))

        print(voc.samples_to_indices(SAMPLES))
        print(voc.sample_to_binary_indices(SAMPLES[2]))
        print('---')
        print(voc.samples_to_binary_indices(SAMPLES))

        print('---')
        print(voc.indices_to_sample(voc.sample_to_indices(SAMPLES[0])))

        print('---')
        print(voc.indices_to_samples(voc.samples_to_indices(SAMPLES)))

        print('---')
        print(voc.binary_indices_to_sample(voc.sample_to_binary_indices(SAMPLES[0])))

        print('---')
        print(voc.binary_indices_to_samples(voc.samples_to_binary_indices(SAMPLES)))
        
        print('---')
        """

        encoded = voc.sample_to_binary_sparse(SAMPLES[0])
        print(encoded)
        print(voc.binary_sparse_to_sample(encoded))

        print('---')

        encoded = voc.samples_to_binary_sparse(SAMPLES)
        print(encoded)
        print(voc.binary_sparse_to_samples(encoded))

        assert True
