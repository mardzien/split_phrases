import pandas as pd
import spacy as sp
import re
import sys
from spacy.lang.pl import Polish

nlp_pl=sp.load("pl_spacy_model_morfeusz_big")