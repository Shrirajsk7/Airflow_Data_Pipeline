#Extracting all the folders from the zip file:
import zipfile

def extract_data():
    with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
        zip_ref.extractall(".")