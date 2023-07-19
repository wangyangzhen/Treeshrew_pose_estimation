# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:11:13 2023

@author: Admion
"""
# "D:/data/behavior/treeshew/movie/demo_movie.mp4"

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog
import cv2
import h5py

class VideoLabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Labeling GUI")
        self.root.geometry("800x600")
        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<space>", self.record_label)

        self.behavior_label = {}
        self.video_label = []

        self.frame_index = 0
        self.label_index = -1

        self.video_path = None
        self.video_capture = None

        self.create_widgets()

    def create_widgets(self):
        self.behavior_label_frame = tk.Frame(self.root)
        self.behavior_label_frame.pack(side=tk.LEFT, padx=10)

        self.behavior_label_label = tk.Label(self.behavior_label_frame, text="Behavior Labels")
        self.behavior_label_label.pack()

        self.behavior_label_text = tk.Text(self.behavior_label_frame, height=20, width=20)
        self.behavior_label_text.pack()

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(side=tk.LEFT, padx=10)

        self.behavior_name_label = tk.Label(self.input_frame, text="Enter Behavior Name:")
        self.behavior_name_label.pack()

        self.behavior_name_entry = tk.Entry(self.input_frame)
        self.behavior_name_entry.pack()

        self.next_button = tk.Button(self.input_frame, text="Next", command=self.record_behavior)
        self.next_button.pack()

        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=10)

        self.select_video_button = tk.Button(self.video_frame, text="Select Video", command=self.select_video)
        self.select_video_button.pack()

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.save_button = tk.Button(self.root, text="Save", command=self.save_labels)
        self.save_button.pack(pady=10)

    def record_behavior(self):
        behavior_name = self.behavior_name_entry.get()
        if behavior_name:
            self.behavior_label[behavior_name] = len(self.behavior_label)
            self.behavior_label_text.insert(tk.END, behavior_name + "\n")
            self.behavior_name_entry.delete(0, tk.END)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.show_frame()

    def show_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))

            self.video_label.image = tk.PhotoImage(data=cv2.imencode(".png", frame)[1].tobytes())
            self.video_label.configure(image=self.video_label.image)

            self.root.after(10, self.show_frame)
        else:
            self.video_capture.release()

    def next_frame(self, event):
        self.record_label()
        self.frame_index += 1
        self.show_frame()

    def prev_frame(self, event):
        if self.frame_index > 0:
            self.frame_index -= 1
            self.show_frame()

    def record_label(self, event=None):
        if self.video_path:
            if self.label_index >= 0:
                self.video_label[self.label_index][1] = self.behavior_label_text.get("1.0", tk.END).strip()

            self.label_index = self.frame_index
            self.video_label.append([self.frame_index, -1])

    def save_labels(self):
        if self.video_path:
            labels_file = self.video_path.replace(".mp4", ".h5")
            with h5py.File(labels_file, "w") as f:
                behavior_label_dataset = f.create_dataset("behavior_label", (len(self.behavior_label),), dtype=h5py.special_dtype(vlen=str))
                behavior_label_dataset[:] = list(self.behavior_label.keys())

                video_label_dataset = f.create_dataset("video_label", (len(self.video_label), 2), dtype=int)
                for i, label in enumerate(self.video_label):
                    video_label_dataset[i] = label

            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoLabelingGUI(root)
    root.mainloop()

