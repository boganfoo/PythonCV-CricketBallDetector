# Initialize Labelbox
import glob
import os

import cv2
from labelbox import Client, Dataset, LabelingFrontend, OntologyBuilder, Tool

api_key = os.getenv("API_KEY")

if api_key is None:
    raise Exception("Please set the API_KEY environment variable")

client = Client(api_key)


# Create a project
project = client.create_project(name="Cricket Ball Detection")

# print information about the project
print(project.uid)
print(project.name)
print(project.created_at)
print(project.updated_at)

dataset = client.create_dataset(name="Cricket Ball Images")
# print information about the dataset
print(dataset.name)
print(dataset.created_at)
print(dataset.updated_at)
print(dataset.uid)


# Associate the dataset with the project
input("any key")

# Define ontology
ontology_builder = OntologyBuilder(
    tools=[Tool(tool=Tool.Type.BBOX, name="Cricket Ball")]
)
labeling_frontend = LabelingFrontend(
    client=client, field_values=ontology_builder.asdict()
)
project.setup(
    labeling_frontend=labeling_frontend, labeling_frontend_options={}
)

# Get a list of all files in the input directory
files = glob.glob("input/*")

# Find the latest file
latest_file = max(files, key=os.path.getctime)

# Use the latest file as the video source
video = cv2.VideoCapture(latest_file)
frame_count = 0
while True:
    success, frame = video.read()
    if not success:
        break

    # Save every 100th frame as an image
    if frame_count % 100 == 0:
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)
        # Upload image to Labelbox
        dataset.create_data_row(row_data=f"frame_{frame_count}.jpg")

    frame_count += 1

video.release()
