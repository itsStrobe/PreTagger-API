import abc
import logging

import numpy as np
import pandas as pd

from FileReadWrite import FileController
from ModelManager import ModelFactory
from PreTaggerEnums import FileDataType, FileType, ProjectType
from PreTaggerKeywords import DataframeKeywords, TagKeywords


class ProjectInterface(metaclass=abc.ABCMeta):

    # - CLASS INDEPENDENT METHODS - #

    @abc.abstractmethod
    def initTagsMask(self):
        """Initialize Tags Mask"""
        raise NotImplementedError

    @abc.abstractmethod
    def extractLabeledAndUnlabeledData(self):
        """Generate Labeled and Unlabeled Data DataFrames"""
        raise NotImplementedError

    @abc.abstractmethod
    def generatePreTags(self):
        """Initialize Model, Train Model, and get Predictions"""
        raise NotImplementedError

    # - CLASS DEPENDENT METHODS - #
    @abc.abstractmethod
    def extractProjectFiles(self, dataDir : str, tagsDir : str):
        """Get Project Data from AWS"""
        raise NotImplementedError


class Project(ProjectInterface):

    def __init__(self, projType : ProjectType):
        # - VARIABLES - #
        self.data = None # Pandas DF containing the project's data
        self.tags = None # Pandas DF containing the project's user-defined tags
        self.pred = None # Pandas DF containing the project's model-defined tags

        self.labeledData = None # Pandas DF containing the project's labeled data (inclued Tag column)
        self.unlabeledData = None # Pandas DF containing the project's unlabeled data

        self.tagsMask = None # Pandas DF mask with True if data was user tagged and False if not.

        self.model = ModelFactory.createModel(projType, {}) # Model to perform training.

    @staticmethod
    def mapValToDf(target : pd.DataFrame, values : [str], col : str, mask : [int]):
        if(target is None or values is None or col is None or mask is None):
            print("One parameter is NoneType")
            raise TypeError

        if(len(values) != len(mask)):
            print("Length of values to be set must be the same as the mask.")
            raise TypeError

        target.loc[mask, col] = values

        return target


    """Set Project Data from Pandas DataFrame"""
    def setProjectData(self, data : pd.DataFrame = None, tags : pd.DataFrame = None):
        if(data is not None):
            self.data = data

        if(tags is not None):
            self.tags = tags

        return True

    """Returns Project Data and Tags DataFrame"""
    def getProjectData(self):
        return self.data, self.tags

    """Returns the Model's Predictions along with the pre-tagged tags."""
    def getPredictions(self):
        return self.pred


    """Initialize Tags Mask. False if content was not tagged; True if it was user tagged."""
    def initTagsMask(self):
        if(self.data is None or self.tags is None):
            print("self.data or self.tags is NoneType")
            raise TypeError

        self.tagsMask = self.tags[DataframeKeywords.FILE_CONTENT_COL] != TagKeywords.UNLABELED_TAG

        logging.info("Tags Mask was created successfully.")


    def extractLabeledAndUnlabeledData(self):
        if(self.data is None or self.tags is None):
            print("self.data or self.tags is NoneType")
            raise TypeError

        self.initTagsMask()

        ## -- LabeledData Creation -- ##
        # Get Labeled Content from the data DF
        labeledData_df = self.data.loc[self.tagsMask == True]
        # Get Tags for Labeled Content from the tags DF
        tags_df = self.tags.loc[self.tagsMask == True]

        # Initialize LabeledData DF with Labeled Content indexing and Data, Tag Columns
        self.labeledData = pd.DataFrame(index=labeledData_df.index, columns=[DataframeKeywords.DATA_COL, DataframeKeywords.TAG_COL])

        # Populate LabeledData DF
        self.labeledData[DataframeKeywords.DATA_COL] = labeledData_df[DataframeKeywords.FILE_CONTENT_COL]
        self.labeledData[DataframeKeywords.TAG_COL] = tags_df[DataframeKeywords.FILE_CONTENT_COL]
        

        ## -- UnlabeledData Creation -- ##
        # Get Unlabeled Content from the data DF
        unlabeledData_df = self.data.loc[self.tagsMask == False]
        
        # Initialize UnlabeledData DF with Unlabeled Content indexing and Data Column
        self.unlabeledData = pd.DataFrame(index=unlabeledData_df.index, columns=[DataframeKeywords.DATA_COL])

        # Populate UnlabeledData DF
        self.unlabeledData[DataframeKeywords.DATA_COL] = unlabeledData_df[DataframeKeywords.FILE_CONTENT_COL]

        logging.info("Successfully Extracted Labeled and Unlabeled Data rows.")

            
    def generatePreTags(self):
        # TODO: Set HyperParameters based on Project Type.

        if(self.labeledData is None and self.unlabeledData is None):
            self.extractLabeledAndUnlabeledData()

        X_clean = self.model.preProcessing(self.labeledData)

        X_clean_unlabeled = self.model.preProcessing(self.unlabeledData)


        self.model.setVocabulary([X_clean, X_clean_unlabeled])
        self.model.setClasses(X_clean)

        self.model.train(X_clean)

        pred_vec = self.model.predict(X_clean_unlabeled)

        self.pred = Project.mapValToDf(self.tags, pred_vec, DataframeKeywords.FILE_CONTENT_COL, list(self.unlabeledData.index))

        return True


class CSVProject(Project):

    def extractProjectFiles(self, dataDir : str, tagsDir : str, fileReader : FileController):

        self.data = fileReader.GetFile(dataDir, asType=FileDataType.DataFrame)

        self.tags = fileReader.GetFile(tagsDir, asType=FileDataType.DataFrame)

        return True

class TXTProject(Project):

    def extractProjectFiles(self, dataDir : str, tagsDir : str, fileReader : FileController):

        self.data = fileReader.GetFiles(dataDir, asType=FileDataType.DataFrame)

        self.tags = fileReader.GetFile(tagsDir, asType=FileDataType.DataFrame)

        return True


class ProjectFactory:

    @staticmethod
    def createProject(fileType : FileType, projType : ProjectType):

        if(fileType == FileType.CSV):
            return CSVProject(projType)
        
        if(fileType == FileType.TXT):
            return TXTProject(projType)

        raise NotImplementedError
