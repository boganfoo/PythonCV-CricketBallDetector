import cv2
from labelbox import Client, LabelingFrontend, OntologyBuilder, Tool

# Initialize Labelbox
with open("key.txt", "r") as file:
    api_key = file.read().strip()

client = Client(api_key)
project = client.create_project(name="Cricket Ball Detection")
dataset = client.create_dataset(name="Cricket Ball Images", projects=project)

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

import glob
import os

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
