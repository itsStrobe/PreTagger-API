#!venv/bin/python3
import os
from flask import Flask, abort, jsonify, request
from threading import Thread

from PreTaggerOrchestrator import PreTaggerOrchestrator
from PreTaggerEnums import FileType, ProjectType
from PreTaggerKeywords import FileKeywords
from config import EnvVariables

# -- API MACROS --
NAME = 'PreTagger'
VER = 'v0.1'

preTagger = PreTaggerOrchestrator(awsBucket=EnvVariables.BUCKET_NAME, awsRegion=EnvVariables.REGION_NAME, awsAccessKeyId=EnvVariables.AWS_ACCESS_KEY_ID, awsSecretAccessKey=EnvVariables.AWS_SECRET_ACCESS_KEY)

app = Flask(__name__)

# -- HELPER METHODS --

def ValidateJSONFields(reqJSON, requiredFields : list == []):

    # Validate JSON Object contains required fields
    missingFields = [field for field in requiredFields if field not in reqJSON]
    if len(missingFields) > 0:
        abort(400, description=f'{missingFields} fields were not found in JSON Body.')

# -- API CALLS -- 

@app.route("/")
def index():
    return "Hello, world!"

@app.route(f"/{NAME}/debug/{VER}/DownloadFromBucket/", methods=['GET'])
def DownloadFromBucket():
    requiredFields = ['fileLoc', 'fileDest']

    print(f"Downloading from Bucket: {EnvVariables.BUCKET_NAME}")

    reqJSON = request.json

    ValidateJSONFields(reqJSON, requiredFields)

    isDownloaded, message = preTagger.DownloadFile(reqJSON['fileLoc'], fileDest=reqJSON['fileDest'])

    if(not isDownloaded):
        abort(400, description=message)

    return message

@app.route(f"/{NAME}/debug/{VER}/UploadToBucket/", methods=['POST'])
def UploadToBucket():
    requiredFields = ['fileLoc', 'fileDest']

    reqJSON = request.json

    ValidateJSONFields(reqJSON, requiredFields)

    isUploaded, message = preTagger.UploadFile(reqJSON['fileLoc'], objectName=reqJSON['fileDest'])

    if(not isUploaded):
        abort(400, description=message)

    return message

@app.route(f"/{NAME}/api/{VER}/Label/", methods=['POST'])
def Label():
    requiredFields = ['userId', 'projectId', 'fileType', 'projectType', 'dataFile', 'tagsFile']

    reqJSON = request.json

    # Validate Request has valid JSON Object.
    if not reqJSON:
        abort(400, description="JSON object not found.")

    ValidateJSONFields(reqJSON, requiredFields)

    print(reqJSON)

    # TODO: Define Target Dir for Tagged Files.

    fileType = None
    projType = None

    if (reqJSON['fileType'] == 'TXT'):
        fileType = FileType.TXT
    elif (reqJSON['fileType'] == 'CSV'):
        fileType = FileType.CSV

    if (reqJSON['projectType'] == "Sentiment Analysis"):
        projType = ProjectType.SENTIMENT_ANALYSIS
    elif (reqJSON['projectType'] == "Text Classification"):
        projType = ProjectType.TEXT_CLASSIFICATION
    elif (reqJSON['projectType'] == "NER Tagging"):
        projType = ProjectType.NER_TAGGING
    elif (reqJSON['projectType'] == "POS Tagging"):
        projType = ProjectType.POS_TAGGING

    projDir = os.path.join(reqJSON['userId'], reqJSON['projectId'])

    dataDir = os.path.join(projDir, reqJSON['dataFile'])
    tagsDir = os.path.join(projDir, reqJSON['tagsFile'])
    predDir = os.path.join(projDir, FileKeywords.SILVER_STANDARD_FILE)

    Thread(target=preTagger.LabelOrchestrator, args=(dataDir, tagsDir, predDir, fileType, projType, )).start()

    resp_data = {
        "message" : f"File with Silver Standard is being created.",
        "silver_standard" : FileKeywords.SILVER_STANDARD_FILE
    }

    return jsonify(resp_data), 202

# -- ERROR HANDLING --

# Bad Request
@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

# Resource Not Found
@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

if __name__ == '__main__':
    app.run(threaded=True, port=EnvVariables.PORT)
