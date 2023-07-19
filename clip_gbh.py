import os
from typing import List
import gradio as gr
import cv2
from PIL import Image
import argparse
import glob
import clip
import torch
import math
import numpy as np
from config import video_paths
import plotly.express as px
from plotly.subplots import make_subplots

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path:str):
    video_frames = []

    # Open the video file
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)

    N=15
    current_frame = 0
    while capture.isOpened():
        ret, frame = capture.read()

        # Convert it to a PIL image (required for CLIP) and store it
        if ret == True:
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break

        # Skip N frames
        current_frame += N
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Print some statistics
    print(f"Frames extracted: {len(video_frames)}")
    return video_frames


def process_frames(frame_list:List[Image.Image]):
    batch_size = 256
    batches = math.ceil(len(frame_list) / batch_size)

    # The encoded features will bs stored in video_features
    video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i+1}/{batches}")
        batch_frames = frame_list[i*batch_size : (i+1)*batch_size]
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)

        # Encode with CLIP and normalize
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

        # Append the batch to the list containing all features
        video_features = torch.cat((video_features, batch_features))

    # Print some stats
    print(f"Features: {video_features.shape}")
    return video_features


def load_frames_for_display(video_filepath, timeframes, max_timeframes=20):
    cap = cv2.VideoCapture(video_filepath)
    if not timeframes:
        return None
    fig = make_subplots(rows=min(len(timeframes), max_timeframes), cols=3)
    for row, timeframe in enumerate(timeframes):
        if row >= max_timeframes:
            break
        start_frame = timeframe['start_frame']
        end_frame = timeframe['end_frame']
    
        for col, i in enumerate([start_frame, (start_frame+end_frame)/2, end_frame]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret == True:
                image = Image.fromarray(frame[:, :, ::-1])
                fig.add_trace(px.imshow(image).data[0], row=row+1, col=col+1)
            else:
                break
    # hide axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    return fig


def load_video_features(video_filepath):
    if not video_filepath:
        return None
    file_basename = video_filepath.split("/")[-1].split(".")[0]
    features_filepath = f"numpy_indices/{file_basename}.npy"
    if os.path.exists(features_filepath):
        # load numpy
        video_features = np.load(features_filepath)
        # if video features are numpy array convert to tensor, convert to numpy
        if isinstance(video_features, torch.Tensor):
            video_features = video_features.cpu().numpy()
    else:
        video_frames = extract_frames(video_filepath)
        video_features = process_frames(video_frames).cpu().numpy()
        # save numpy
        np.save(features_filepath, video_features)
    return torch.from_numpy(video_features)


def frame_to_time(frame_number):
    """
    Converts frames to seconds for time-based annotations
    :param frame_number:
    :return:
    """
    return 1000 * frame_number / 29.97


def search_video(text_query, image_query, video_filepath, threshold, k_results=5):
    video_features = load_video_features(video_filepath)
    
    if image_query is None:
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize(text_query).to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        features = text_features
    else:
        with torch.no_grad():
            image = preprocess(Image.fromarray(image_query)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        features = image_features
    features = features.cpu().numpy()
    # Compute the similarity between the search query and each frame using the Cosine similarity
    similarities = video_features @ features.T
    # values, best_photo_idx = similarities.topk(k_results, dim=0)

    fig = px.imshow(similarities.T.cpu().numpy().tolist(), height=50, aspect='auto', color_continuous_scale='viridis')
    fig.update_layout(coloraxis_showscale=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    # Find the frames that meet the threshold
    above_threshold_indices = np.where(similarities > threshold)[0]
    timeframes = []
    if len(above_threshold_indices) > 0:
        # Find the contiguous regions of frames above the threshold
        contiguous_regions = np.split(above_threshold_indices,
                                        np.where(np.diff(above_threshold_indices) != 1)[0] + 1)

        # For each contiguous region find the start and end times
        for region in contiguous_regions:
            start_frame, end_frame = int(region[0]) * 15 , int(region[-1]) * 15
            start_time = frame_to_time(start_frame)
            end_time = frame_to_time(end_frame)

            timeframe = {'start': start_time, 'end': end_time,
                            'start_frame': start_frame, 'end_frame': end_frame}
            timeframes.append(timeframe)
    images = load_frames_for_display(video_filepath, timeframes)
    return fig, timeframes, video_filepath, images


with gr.Blocks() as demo:
    with gr.Tab("Text Query"):
        text_query = gr.Textbox(lines=2, value="rolling credits", label="Search query")
    with gr.Tab("Image Query"):
        image_query = gr.Image()

    inp = gr.Dropdown(video_paths, value="/data/clams/wgbh/NewsHour/cpb-aacip-507-cf9j38m509.mp4", label="Select a video")
    search_button = gr.Button(label="Search")
    #video
    video = gr.Video(inp.value)
    #heatmap
    heatmap = gr.Plot()
    threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Threshold")
    with gr.Accordion("Time Segments", open=False):
        segments = gr.JSON()
    with gr.Accordion("Images", open=True):
        images = gr.Plot()
    demo.load(search_video, inputs=[text_query, image_query, inp, threshold], outputs=[heatmap, segments, video, images])

    # search button
    search_button.click(search_video, inputs=[text_query, image_query, inp, threshold], outputs=[heatmap, segments, video, images])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="User")
    args, unknown = parser.parse_known_args()
    demo.launch()


