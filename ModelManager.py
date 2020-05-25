import abc
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from PreTaggerEnums import ProjectType
from PreTaggerKeywords import DataframeKeywords


class ModelInterface(metaclass=abc.ABCMeta):

    ## -- CLASS INDEPENDENT METHODS -- ##

    @abc.abstractstaticmethod
    def preProcessing(X : pd.DataFrame, dataCol : str):
        """ Pre-process the Text by applying common NLP techniques dependent on the model type."""
        raise NotImplementedError

    ## -- CLASS DEPENDENT METHODS -- ##

    @abc.abstractmethod
    def setVocabulary(self, X : [pd.DataFrame]):
        """ Set Model's Vocabulary. This will typically initialize the Vectorizer object. """
        raise NotImplementedError

    @abc.abstractmethod
    def setClasses(self, y : [str]):
        """ Set Model's Classes. This ensures that all classes are represented """
        raise NotImplementedError

    @abc.abstractmethod
    def setHyperParam(self, hyperParams : {}):
        """ Set Model's HyperParameters """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL, tagCol : str = DataframeKeywords.TAG_COL):
        """ Train the Model """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X : pd.DataFrame):
        """ Get Predictions """
        raise NotImplementedError


class Model(ModelInterface):

    @staticmethod
    def preProcessing(X : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL):

        for _, row in X.iterrows():
            # Remove capitalization
            row[dataCol] = row[dataCol].lower()

            # Remove punctuation
            row[dataCol] = row[dataCol].translate(str.maketrans("", "", string.punctuation))

            # Remove whitespaces
            row[dataCol] = row[dataCol].strip()

        return X


class TextClassificationModel(Model):

    def __init__(self, hyperParams : {}):
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        self.classEncoder = LabelEncoder()

        # self.setHyperParam(hyperParams)

    def setVocabulary(self, X : [pd.DataFrame], dataCol : str = DataframeKeywords.DATA_COL):
        X_vec = []

        for text in X:
            X_vec.extend(text[dataCol].values)

        self.vectorizer.fit(X_vec)

    def setClasses(self, y : pd.DataFrame, tagCol : str = DataframeKeywords.TAG_COL):
        y_vec = list(y[tagCol].values)

        self.classEncoder.fit(y_vec)

    # TODO: Fix bugs
    def setHyperParam(self, hyperParams : {}):
        self.clf.set_params(hyperParams)

    def train(self, data : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL, tagCol : str = DataframeKeywords.TAG_COL):
        if(self.vectorizer is None):
            self.setVocabulary([data])

        if(self.classEncoder is None):
            self.setClasses(data[DataframeKeywords.TAG_COL].values)

        X_enc = self.vectorizer.transform(data[dataCol].values)
        y_enc = self.classEncoder.transform(data[tagCol].values)

        self.clf.fit(X_enc, y_enc)

    def predict(self, X : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL):
        if(self.vectorizer is None or self.classEncoder is None):
            print("Vectorizer or ClassEncoder not initialized.")
            raise TypeError

        X_enc = self.vectorizer.transform(X[dataCol].values)

        y_enc = self.clf.predict(X_enc)

        return list(self.classEncoder.inverse_transform(y_enc))

class SentimentAnalysisModel(Model):

    def __init__(self, hyperParams : {}):
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        self.classEncoder = LabelEncoder()

        # self.setHyperParam(hyperParams)

    def setVocabulary(self, X : [pd.DataFrame], dataCol : str = DataframeKeywords.DATA_COL):
        X_vec = []

        for text in X:
            X_vec.extend(text[dataCol].values)

        self.vectorizer.fit(X_vec)

    def setClasses(self, y : pd.DataFrame, tagCol : str = DataframeKeywords.TAG_COL):
        y_vec = list(y[tagCol].values)

        self.classEncoder.fit(y_vec)

    # TODO: Fix bugs
    def setHyperParam(self, hyperParams : {}):
        self.clf.set_params(hyperParams)

    def train(self, data : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL, tagCol : str = DataframeKeywords.TAG_COL):
        if(self.vectorizer is None):
            self.setVocabulary([data])

        if(self.classEncoder is None):
            self.setClasses(data[DataframeKeywords.TAG_COL].values)

        X_enc = self.vectorizer.transform(data[dataCol].values)
        y_enc = self.classEncoder.transform(data[tagCol].values)

        self.clf.fit(X_enc, y_enc)

    def predict(self, X : pd.DataFrame, dataCol : str = DataframeKeywords.DATA_COL):
        if(self.vectorizer is None or self.classEncoder is None):
            print("Vectorizer or ClassEncoder not initialized.")
            raise TypeError

        X_enc = self.vectorizer.transform(X[dataCol].values)

        y_enc = self.clf.predict(X_enc)

        return list(self.classEncoder.inverse_transform(y_enc))

class ModelFactory:

    @staticmethod
    def createModel(projType : ProjectType, hyperParams : {}):

        if(projType == ProjectType.TEXT_CLASSIFICATION):
            return TextClassificationModel(hyperParams)

        if(projType == ProjectType.SENTIMENT_ANALYSIS):
            return SentimentAnalysisModel(hyperParams)

        raise NotImplementedError
