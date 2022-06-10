# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
import spacy


def cosine_sim(text1, text2):
    """requires    python -m spacy download en_core_web_lg   for slightly better similarity values,
    or simply   python -m spacy download en_core_web_sm   for smaller instalation download (12MB vs 791MB)"""

    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    similarity = doc1.similarity(doc2)
    return similarity
