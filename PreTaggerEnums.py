from enum import Enum

class FileType(Enum):
    CSV = 1
    TXT = 2

class ProjectType(Enum):
    SENTIMENT_ANALYSIS = 1
    TEXT_CLASSIFICATION = 2
    NER_TAGGING = 3
    POS_TAGGING = 4

class FileDataType(Enum):
    List = 1 # List of rows
    DataFrame = 2   # Pandas DataFrame
    Numpy = 3   # Numpy Array
    Text = 4 # String
