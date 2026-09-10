# -*- coding: utf-8 -*-
"""
app.py

Gradio inference application for Wildlife-MoE.

    models/precision_expert/best.pt
    models/kinetic_expert/best.pt

Run:
    python inference/app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import RTDETR, YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    PRECISION_MODEL_PATH,
    KINETIC_MODEL_PATH,
    MOE_THRESHOLD_DEFAULT,
    INFERENCE_CONFIDENCE,
    TRACKER_CONFIG,
)

for path, label in [(PRECISION_MODEL_PATH, "Precision"), (KINETIC_MODEL_PATH, "Kinetic")]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} Expert checkpoint not found at {path}. "
            "Train the models first with training/train.py, or set the "
            "PRECISION_MODEL_PATH / KINETIC_MODEL_PATH environment "
            "variables to point at existing checkpoints."
        )

expert_p = RTDETR(str(PRECISION_MODEL_PATH))
expert_k = YOLO(str(KINETIC_MODEL_PATH))

custom_css = """
body { background-color: #05070a; overflow: hidden; }
.gradio-container { height: 100vh !important; }
.video-display { height: 530px !important; }
.metric-val { color: #10b981; font-weight: bold; font-size: 1.1em; }
"""


def process_video_v3(video_in, mode, threshold, progress=gr.Progress()):
    if video_in is None:
        return None

    cap = cv2.VideoCapture(video_in)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

        if mode == "Option 3: MoE (Auto)":
            active_model = expert_k if score < threshold else expert_p
            expert_tag = "Kinetic" if score < threshold else "Precision"
        else:
            active_model = expert_p if "Precision" in mode else expert_k
            expert_tag = "Manual"

        results = active_model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=INFERENCE_CONFIDENCE,
            verbose=False,
        )[0]

        annotated = results.plot()

        cv2.rectangle(annotated, (0, 0), (width, 25), (255, 255, 255), -1)
        hud_text = f"TRACKING ID: {expert_tag} | SENSOR QUALITY: {int(score)}"
        cv2.putText(annotated, hud_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out.write(annotated)
        frame_count += 1
        progress(frame_count / total_frames, desc="Tracking Animals...")

    cap.release()
    out.release()
    return temp_out.name


def build_demo():
    with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="emerald")) as demo:
        gr.HTML(
            "<h2 style='color: #10b981; margin: 10px 0; text-align: center;'>"
            "WILDLIFE TRACKING STATION</h2>"
        )

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                input_video = gr.Video(label="Source", height=280)
                mode_sel = gr.Dropdown(
                    ["Option 3: MoE (Auto)", "Precision Only", "Kinetic Only"],
                    label="MODE",
                    value="Option 3: MoE (Auto)",
                )
                q_thresh = gr.Slider(50, 250, value=MOE_THRESHOLD_DEFAULT, label="THRESHOLD")
                run_btn = gr.Button("🚀 START TRACKING", variant="primary")

            with gr.Column(scale=3):
                output_video = gr.Video(
                    label="TRACKED OUTPUT", interactive=False, elem_classes="video-display"
                )

        run_btn.click(
            fn=process_video_v3,
            inputs=[input_video, mode_sel, q_thresh],
            outputs=output_video,
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
