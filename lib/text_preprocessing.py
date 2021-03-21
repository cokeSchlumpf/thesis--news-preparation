import re
import spacy

from functional import seq
from spacy.tokens import Token
from spacy.language import Language
from typing import Callable, List, Optional

Token.set_extension('ppt_output', default='')
WS_PATTERN = re.compile(r'\s{2,}')


def lemmatize(token: Token) -> None:
    """
    Use the lemmatized version of a word.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = token.lemma_


def remove_stopwords(token: Token) -> None:
    """
    Deselects stopwords from the text.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    if token.is_stop:
        token._.ppt_output = token.whitespace_


def simple_punctuation_only(token: Token) -> None:
    """
    Deselects tokens which are not alpha-numeric or ',', '.', '?', '!', ':', ';', '-', '(' or ')'

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    cond = len(token.text) == 1 and not (token.is_alpha or token.is_digit or token.text in [',', '.', '?', '!', '(', ')', ':', ';', '-'])

    if cond:
        token._.ppt_output = token.whitespace_


def words_only(token: Token) -> None:
    """
    Deselects every token which is not a word.

    :param token: The spaCy token
    :return Nothing; Token is mutated
    :return:
    """
    if not token.is_alpha:
        token._.ppt_output = token.whitespace_


def to_lower(token: Token) -> None:
    """
    Turns all characters to lower case.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = token._.ppt_output.lower()


def tokenize(token: Token) -> None:
    """
    Will ensure that every token is separated by a whitespace.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = f"{token._.ppt_output} "


def use_text(token: Token) -> None:
    """
    Sets the text of the token as the expected output text. This step must be included in any pipeline.

    :param token: The spaCy token
    :return: Nothing; Token is mutated
    """
    token._.ppt_output = token.text_with_ws


def preprocess_text(
        s: str, lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None) -> str:
    """
    Preprocesses a string with the help of spaCy. Allows different composable pre-processing methods.

    :param s: The string to be pre-processed
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :return: The pre-processed string
    """
    doc, pipeline = _parse(s, lang, pipeline)

    for task in pipeline:
        for token in doc:
            task(token)

    joined = ''.join([token._.ppt_output for token in doc])
    return re.sub(WS_PATTERN, ' ', joined)


def preprocess_tokens(
        s: str, lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None) -> List[str]:

    """
    Preprocesses a string with the help of spaCy. Allows different composable pre-processing methods.

    :param s: The string to be pre-processed
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :return: The pre-processed list of tokens
    """

    doc, pipeline = _parse(s, lang, pipeline)

    for task in pipeline:
        for token in doc:
            task(token)

    return [str.strip(token._.ppt_output) for token in doc if len(str.strip(token._.ppt_output)) > 0]


def preprocess_sentences(
        s: str, lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None) -> str:
    """
    Just like preprocess_text, just that this function returns a list of sentences.

    :param s: The string to be pre-processed
    :param lang: A spaCy language pipeline. As returned by `spacy.load()`
    :param pipeline:  A list of pre-processing functions
    :return: The pre-processed string
    """

    doc, pipeline = _parse(s, lang, pipeline)

    for task in pipeline:
        for token in doc:
            task(token)

    return seq(doc.sents)\
        .map(lambda sentence: ''.join([token._.ppt_output for token in sentence]))\
        .map(lambda sentence: re.sub(WS_PATTERN, ' ', sentence))


def _parse(
        s: str, lang: Optional[Language] = None,
        pipeline: Optional[List[Callable[[Token], None]]] = None):
    if lang is None:
        lang = spacy.load('de_dep_news_trf')

    if pipeline is None:
        pipeline = [to_lower, simple_punctuation_only]

    pipeline = [use_text] + pipeline
    return lang(s), pipeline


if __name__ == '__main__':
    print(preprocess_text('Hello "World" und so!'))
