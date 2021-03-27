import multiprocessing
import spacy

from collections import defaultdict
from functional import seq
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from joblib import Memory
from lib import data
from lib.utils import root_directory
from lib.text_preprocessing import preprocess_text, preprocess_all, remove_stopwords, to_lower, tokenize, words_only
from tqdm import tqdm

mem = Memory('../data/cache', verbose=0)
root_dir = root_directory()


@mem.cache
def load_data():
    return data.load_data(base_dir=f"{root_dir}/data")


@mem.cache
def compute_text_preprocessing(df):
    preprocess_pipeline = [to_lower, words_only, remove_stopwords, tokenize]
    lang = spacy.load('de_dep_news_trf', disable=['tagger', 'morphologizer', 'attribute_ruler'])

    return seq(tqdm(df['text'].to_numpy()))\
        .map(lambda text: preprocess_text(text, lang, preprocess_pipeline))\
        .to_list()


@mem.cache
def compute_sentences(df):
    sent = [row.split() for row in df['cleaned']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)

    bigram = Phraser(phrases)
    sentences = bigram[sent]

    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1

    print(len(word_freq))
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

    return sentences


@mem.cache
def compute_embedding(texts):
    w2v_model = Word2Vec(
        min_count=20,
        window=4,
        vector_size=300,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1)

    print("Creating vocabulary")
    w2v_model.build_vocab(texts, progress_per=1000)

    print("Train embedding")
    w2v_model.train(texts, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    return w2v_model


if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    samples = load_data()
    samples = samples[samples['lang'] == 'de']
    samples['cleaned'] = compute_text_preprocessing(samples)
    sentences = compute_sentences(samples)
    embedding = compute_embedding(sentences)
    print(embedding)
