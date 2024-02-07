import cv2
import tkinter as tk
from PIL import Image, ImageTk
from object_detection import object_detection_pipeline, load_object_detection_model, load_lane_detection_model, road_lines

class VideoApp:
    def __init__(self, window, window_title, input_video_path, output_video_path):
        self.window = window
        self.window.title(window_title)

        # Load the object detection model
        self.model = load_object_detection_model()
        self.lanemodel = load_lane_detection_model()

        self.video_source = cv2.VideoCapture(input_video_path)
        self.out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), int(self.video_source.get(5)), (int(self.video_source.get(3)), int(self.video_source.get(4))))

        self.canvas = tk.Canvas(window, width=self.video_source.get(3), height=self.video_source.get(4))
        self.canvas.pack()

        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.on_quit)
        self.btn_quit.pack(padx=20, pady=10)

        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.video_source.read()
        if ret:
            processed_frame = object_detection_pipeline(frame, self.model)
            processed_frame = cv2.resize(processed_frame, (1280, 720))
            processed_frame = road_lines(processed_frame, self.lanemodel)
            self.photo = self.convert_to_tk_image(processed_frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.out.write(processed_frame)

        self.window.after(10, self.update)

    def convert_to_tk_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)
        return photo

    def on_quit(self):
        self.video_source.release()
        self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    # Specify your input and output video paths
    input_video_path = 'testvideos/trynew13.mp4'
    output_video_path = 'newvideo.mp4'

    # Create the Tkinter window
    root = tk.Tk()
    app = VideoApp(root, "Video Processing App", input_video_path, output_video_path)
