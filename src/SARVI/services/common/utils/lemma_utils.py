## Copied from https://github.com/lcampillos/MedLexSp

import stanza

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SARVI.models.schemas import PipelineContext

from SARVI.data_io.reader import(
    read_pickle_single
)

###
POSData = read_pickle_single(Path(__file__).resolve().parents[5] / "docs/MedLexSp_v2/stanza_lemmatizer/MedLexSpPOS.pickle")
###

# Helper functions to change label names and format
def format_pos_name(POS_label_name,predicted_POS):
    """
    Given the name of a part-of-speech tag in MedLexSp dictionary, changes the tag name according to Universal Dependencies used in Spacy / Stanza. 
    In case several tags are possible, the part-of-speech prediction is used to disambigate. 
    E.g. "ADJ;N" (tag name in MedLexSp) and "ADJ" (predicted tag) => output "ADJ". 
    E.g. "N;NPR" -> "NOUN" or "PROPN", "ADJ;N" -> "ADJ" or "NOUN"
    MedLexSp category "AFF" has not an equivalent category in Spacy / Stanza.

    Parameters
    ----------
        `POS_label_name`: Any
            - Entity or relation labels used by the model.
        `predicted_POS`: Any
            - Argument controlling predicted pos.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    # keys are MedLexSp PoS codes, values are Spacy / Stanza labels
    POSFormat = {'ADJ': 'ADJ', 'ADV': 'ADV', 'N': 'NOUN', 'PREP': 'ADP', 'V': 'VERB', 'art': 'DET', 'NPR': 'PROPN'}

    if ((POS_label_name == "ADJ;ADV") and (predicted_POS == "ADV")):
        return POSFormat['ADV']
    elif ((POS_label_name == "ADJ;ADV") and (predicted_POS == "ADJ")):
        return POSFormat['ADJ']
    elif ((POS_label_name == "N;NPR") and (predicted_POS == "NOUN")):
        return POSFormat['N']
    elif ((POS_label_name == "N;NPR") and (predicted_POS == "PROPN")):
        return POSFormat['NPR']
    elif ((POS_label_name == "ADJ;N") and (predicted_POS == "ADJ")):
        return POSFormat['ADJ']
    elif ((POS_label_name == "ADJ;N") and (predicted_POS == "NOUN")):
        return POSFormat['N']
    else:
        return POSFormat[POS_label_name]


def get_pos_from_lexicon(word,predicted_POS,POSDict):

    """
    Looks up the part-of-speech category for a word in the lexicon.

    Parameters
    ----------
        `word`: Any
            - Argument controlling word.
        `predicted_POS`: Any
            - Argument controlling predicted pos.
        `POSDict`: Any
            - Argument controlling posdict.

    Returns
    -------
        `Any`
            - Matching value or collection, when available.
    """

    try:
        word = word.lower()
        if POSDict[word]:
            TuplesList = POSDict[word]
            # Look up the dictionary using the PoS tag, if several categories are possible: "curva": [('ADJ', 'curvo'), ('N', 'curva')]
            if len(TuplesList)>1:
                # Default value (in case the following step fails)
                POS = TuplesList[0][0]
                lemma = TuplesList[0][1]
                # Take the lemma according to PoS predicted by Stanza/Spacy
                for Tuple in TuplesList:
                    POS = Tuple[0]
                    if format_pos_name(POS,predicted_POS) == predicted_POS:
                        lemma = Tuple[1]
                        return format_pos_name(POS,predicted_POS), lemma
            else:
                POS=TuplesList[0][0]
                lemma=TuplesList[0][1]

            return format_pos_name(POS,predicted_POS),lemma
    except:
        return None

def load_lemmatizer(device: str):
    """
    Loads the Stanza lemmatizer on the requested device.

    Parameters
    ----------
        `device`: str
            - Device used for tensor computation.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    lemmatizer = stanza.Pipeline('es', processors='tokenize,pos,lemma', verbose=False, use_gpu=device == "cuda", download_method=None, 
                                 lemma_model_path=str(Path(__file__).resolve().parents[5] / "docs/MedLexSp_v2/stanza_lemmatizer/ancora-medlexsp-stanza112.pt"), 
                                 lemma_forward_charlm_path="", lemma_backward_charlm_path="")
    return lemmatizer

def run_lemmatizer(lemmatizer: stanza.Pipeline, text: str):
    """
    Lemmatizes the supplied text with the configured Stanza pipeline.

    Parameters
    ----------
        `lemmatizer`: stanza.Pipeline
            - Argument controlling lemmatizer.
        `text`: str
            - Text containing the entity or span.

    Returns
    -------
        `Any`
            - Processed output for downstream pipeline steps.
    """
    doc = lemmatizer(text)

    lemmatized_tokens = []
    token_data = []

    for sentence in doc.sentences:
        for word in sentence.words:
            original_text = word.text
            pos = word.pos
            lemma = word.lemma

            lexicon_result = get_pos_from_lexicon(original_text.lower(), pos, POSData)

            if lexicon_result:
                pos, lemma = lexicon_result

            lemmatized_tokens.append(lemma)

            token_data.append({"text": original_text, "lemma": lemma, "pos": pos})

    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text, token_data