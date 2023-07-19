# Video Search Application with CLIP and Gradio

This application allows you to search videos using either text or image queries. The search is powered by OpenAI's CLIP model and implemented in Python using the Gradio library.

## Requirements

You need to have these installed:
- Python 3.9 or later

You can install the required packages using pip:
```
pip install -r requirements.txt
```

## How to Run

Run the python script from the command line:

```
gradio clip_gbh.py
```
Modify the filepath in the config.py file to match your data directory.
## Usage

Upon launching the application, you will see an interface with options to search a video using either a text query or an image query.

1. **Text Query Tab:** Enter a text query and click on the "Search" button. The application will display a heatmap showing the similarities of each frame of the selected video to the text query.

2. **Image Query Tab:** Upload an image and click on the "Search" button. The application will display a heatmap showing the similarities of each frame of the selected video to the image query.

You can adjust the threshold for the similarity score using the slider. Regions with a similarity score above the threshold will be considered as potential matches.

The application will also display the time segments and images of the frames that match the query based on the threshold. The time segments will be displayed in milliseconds.

Note: For performance reasons, the application extracts every 15th frame from the video for processing. Thus, the time segments are not continuous but based on these sampled frames.

## Troubleshooting

If you encounter any issues while using the application, ensure you have all the required packages installed and that the video files are located in the correct directory. If the issue persists, try adjusting the parameters or the type of query you're using.