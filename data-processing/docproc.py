import os
from typing import Iterator, Sequence
import re
from PyPDF2 import PdfReader
from spacy import Language
from spacy.tokens import Span


def get_pdf_files(path: str) -> list[str]:
    if not os.path.isdir(path):
        raise NotADirectoryError
    return [x for x in os.listdir(path) if x.endswith('.pdf')]


def load_file(path: str) -> PdfReader:
    return PdfReader(path) if path.endswith('.pdf') else None


def load_folder(path: str) -> dict[str: PdfReader]:
    return {x: PdfReader(os.path.join(path, x)) for x in get_pdf_files(path)}


def get_document_text(reader: PdfReader) -> str:
    return ''.join([pg.extract_text() for pg in reader.pages])


def extract_entities(nlp: Language, text: str) -> Sequence[Span]:
    return nlp(text).ents


def get_sentences(nlp: Language, text: str) -> Iterator[Span]:
    return nlp(text).sents


def extract_emails(nlp: Language, text: str) -> list[str]:
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    return [token.text.strip() for token in nlp(text) if email_pattern.match(token.text)]


def extract_dates(nlp: Language, text: str) -> list[Span]:
    return [ent for ent in nlp(text).ents if ent.label_ == 'DATE']


def extract_features_from_sentence(nlp: Language, sentence: str) -> dict[str, list]:
    dates = extract_dates(nlp, sentence)
    emails = extract_emails(nlp, sentence)
    return {
        'dates': dates,
        'email_addresses': emails,
        'text': sentence
    }


def extract_features_from_pdf(nlp: Language, path: str) -> list[dict[str, list]]:
    reader = load_file(path)
    if reader is None:
        raise ValueError('Unable to load file.')

    document_text = get_document_text(reader)
    sentences = get_sentences(nlp, document_text)

    return [extract_features_from_sentence(nlp, sentence.text) for sentence in sentences]
