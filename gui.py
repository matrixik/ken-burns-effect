#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import subprocess
import threading
import cv2
from PIL import Image, ImageTk
import sys


class KenBurnsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ken Burns Effect Generator")
        
        # Start maximized - cross-platform approach
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                # Fallback to manual maximization
                self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")

        # Variables for configuration
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="./images/kbe/")
        self.estim_path = tk.StringVar(
            value="./models/trained/disparity-estimation-mask.tar"
        )
        self.refine_path = tk.StringVar(
            value="./models/trained/disparity-refinement.tar"
        )
        self.inpaint_path = tk.StringVar(value="./models/trained/inpainting-color.tar")

        # Ken Burns parameters
        self.start_u = tk.IntVar(value=512)
        self.start_v = tk.IntVar(value=512)
        self.end_u = tk.IntVar(value=600)
        self.end_v = tk.IntVar(value=600)
        self.start_w = tk.IntVar(value=400)
        self.start_h = tk.IntVar(value=200)
        self.end_w = tk.IntVar(value=300)
        self.end_h = tk.IntVar(value=150)

        # Options
        self.write_frames = tk.BooleanVar(value=True)
        self.dolly = tk.BooleanVar(value=False)
        self.twod = tk.BooleanVar(value=False)

        # Current process
        self.current_process = None
        
        # Video playback variables
        self.video_cap = None
        self.video_playing = False
        self.video_paused = False
        self.video_fps = 30
        self.current_frame = 0
        self.total_frames = 0
        self.video_after_id = None

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Configuration
        left_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right panel - Video display
        right_frame = ttk.LabelFrame(main_frame, text="Video Preview", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Bottom panel - Command output
        bottom_frame = ttk.LabelFrame(self.root, text="Command Output", padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        self.setup_config_panel(left_frame)
        self.setup_video_panel(right_frame)
        self.setup_output_panel(bottom_frame)

    def setup_config_panel(self, parent):
        # File selection
        ttk.Label(parent, text="Input Image:").pack(anchor=tk.W, pady=(0, 5))
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_path, width=30).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        ttk.Label(parent, text="Output Directory:").pack(anchor=tk.W, pady=(10, 5))
        output_frame = ttk.Frame(parent)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_path, width=30).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        # Model paths
        ttk.Label(parent, text="Model Paths:").pack(anchor=tk.W, pady=(10, 5))

        ttk.Label(parent, text="Estimation Model:").pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=self.estim_path, width=40).pack(
            fill=tk.X, pady=(0, 5)
        )

        ttk.Label(parent, text="Refinement Model:").pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=self.refine_path, width=40).pack(
            fill=tk.X, pady=(0, 5)
        )

        ttk.Label(parent, text="Inpainting Model:").pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=self.inpaint_path, width=40).pack(
            fill=tk.X, pady=(0, 10)
        )

        # Ken Burns parameters
        ttk.Label(parent, text="Ken Burns Parameters:").pack(anchor=tk.W, pady=(10, 5))

        params_frame = ttk.Frame(parent)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Start position
        start_frame = ttk.LabelFrame(params_frame, text="Start Position")
        start_frame.pack(fill=tk.X, pady=(0, 5))

        start_pos_frame = ttk.Frame(start_frame)
        start_pos_frame.pack(fill=tk.X)
        ttk.Label(start_pos_frame, text="U:").pack(side=tk.LEFT)
        ttk.Entry(start_pos_frame, textvariable=self.start_u, width=8).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(start_pos_frame, text="V:").pack(side=tk.LEFT)
        ttk.Entry(start_pos_frame, textvariable=self.start_v, width=8).pack(
            side=tk.LEFT
        )

        start_size_frame = ttk.Frame(start_frame)
        start_size_frame.pack(fill=tk.X)
        ttk.Label(start_size_frame, text="W:").pack(side=tk.LEFT)
        ttk.Entry(start_size_frame, textvariable=self.start_w, width=8).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(start_size_frame, text="H:").pack(side=tk.LEFT)
        ttk.Entry(start_size_frame, textvariable=self.start_h, width=8).pack(
            side=tk.LEFT
        )

        # End position
        end_frame = ttk.LabelFrame(params_frame, text="End Position")
        end_frame.pack(fill=tk.X, pady=(5, 0))

        end_pos_frame = ttk.Frame(end_frame)
        end_pos_frame.pack(fill=tk.X)
        ttk.Label(end_pos_frame, text="U:").pack(side=tk.LEFT)
        ttk.Entry(end_pos_frame, textvariable=self.end_u, width=8).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(end_pos_frame, text="V:").pack(side=tk.LEFT)
        ttk.Entry(end_pos_frame, textvariable=self.end_v, width=8).pack(side=tk.LEFT)

        end_size_frame = ttk.Frame(end_frame)
        end_size_frame.pack(fill=tk.X)
        ttk.Label(end_size_frame, text="W:").pack(side=tk.LEFT)
        ttk.Entry(end_size_frame, textvariable=self.end_w, width=8).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(end_size_frame, text="H:").pack(side=tk.LEFT)
        ttk.Entry(end_size_frame, textvariable=self.end_h, width=8).pack(side=tk.LEFT)

        # Options
        ttk.Label(parent, text="Options:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Checkbutton(parent, text="Write Frames", variable=self.write_frames).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(parent, text="Dolly Zoom", variable=self.dolly).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(parent, text="2D Mode", variable=self.twod).pack(anchor=tk.W)

        # Preset buttons
        ttk.Label(parent, text="Quick Presets:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Button(parent, text="Half-size Crop", command=self.preset_half_crop).pack(
            fill=tk.X, pady=(0, 5)
        )
        ttk.Button(parent, text="Zoom In", command=self.preset_zoom_in).pack(
            fill=tk.X, pady=(0, 5)
        )
        ttk.Button(parent, text="Pan Right", command=self.preset_pan_right).pack(
            fill=tk.X, pady=(0, 10)
        )

        # Generate button
        self.generate_btn = ttk.Button(
            parent, text="Generate Ken Burns Effect", command=self.generate_effect
        )
        self.generate_btn.pack(fill=tk.X, pady=(10, 0))

        # Stop button
        self.stop_btn = ttk.Button(
            parent,
            text="Stop Generation",
            command=self.stop_generation,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(fill=tk.X, pady=(5, 0))

    def setup_video_panel(self, parent):
        # Video canvas
        self.video_canvas = tk.Canvas(parent, bg="black", width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Video controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X)

        ttk.Button(controls_frame, text="Load Video", command=self.load_video).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.play_pause_btn = ttk.Button(controls_frame, text="Play", command=self.toggle_video)
        self.play_pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(controls_frame, text="Stop", command=self.stop_video).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Progress info
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.progress_var).pack(side=tk.RIGHT)
        
        # Video progress bar
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.video_progress = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video)
        self.video_progress.pack(fill=tk.X, padx=(0, 10), side=tk.LEFT, expand=True)
        
        self.time_label = ttk.Label(progress_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT)

    def setup_output_panel(self, parent):
        # Command output text area
        self.output_text = scrolledtext.ScrolledText(parent, height=8, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.input_path.set(filename)
            self.auto_set_parameters()

    def browse_output(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_path.set(dirname)

    def auto_set_parameters(self):
        """Automatically set parameters based on input image size"""
        if self.input_path.get():
            try:
                img = cv2.imread(self.input_path.get())
                if img is not None:
                    h, w = img.shape[:2]
                    # Set center coordinates
                    center_u, center_v = w // 2, h // 2
                    self.start_u.set(center_u)
                    self.start_v.set(center_v)
                    self.end_u.set(center_u)
                    self.end_v.set(center_v)

                    # Set crop sizes (start with full, end with 3/4)
                    self.start_w.set(w)
                    self.start_h.set(h)
                    self.end_w.set(int(w * 0.75))
                    self.end_h.set(int(h * 0.75))

                    self.log_output(f"Auto-configured for image size: {w}x{h}")
            except Exception as e:
                self.log_output(f"Error reading image: {e}")

    def preset_half_crop(self):
        """Set parameters for half-size cropping effect"""
        if self.input_path.get():
            try:
                img = cv2.imread(self.input_path.get())
                if img is not None:
                    h, w = img.shape[:2]
                    center_u, center_v = w // 2, h // 2

                    self.start_u.set(center_u)
                    self.start_v.set(center_v)
                    self.end_u.set(center_u)
                    self.end_v.set(center_v)

                    self.start_w.set(w)
                    self.start_h.set(h)
                    self.end_w.set(w // 2)
                    self.end_h.set(h // 2)

                    self.log_output("Applied half-size crop preset")
            except Exception as e:
                self.log_output(f"Error applying preset: {e}")

    def preset_zoom_in(self):
        """Set parameters for zoom-in effect"""
        if self.input_path.get():
            try:
                img = cv2.imread(self.input_path.get())
                if img is not None:
                    h, w = img.shape[:2]
                    center_u, center_v = w // 2, h // 2

                    self.start_u.set(center_u)
                    self.start_v.set(center_v)
                    self.end_u.set(center_u)
                    self.end_v.set(center_v)

                    self.start_w.set(int(w * 0.9))
                    self.start_h.set(int(h * 0.9))
                    self.end_w.set(int(w * 0.6))
                    self.end_h.set(int(h * 0.6))

                    self.log_output("Applied zoom-in preset")
            except Exception as e:
                self.log_output(f"Error applying preset: {e}")

    def preset_pan_right(self):
        """Set parameters for pan-right effect"""
        if self.input_path.get():
            try:
                img = cv2.imread(self.input_path.get())
                if img is not None:
                    h, w = img.shape[:2]

                    self.start_u.set(int(w * 0.3))
                    self.start_v.set(h // 2)
                    self.end_u.set(int(w * 0.7))
                    self.end_v.set(h // 2)

                    crop_w = int(w * 0.8)
                    crop_h = int(h * 0.8)
                    self.start_w.set(crop_w)
                    self.start_h.set(crop_h)
                    self.end_w.set(crop_w)
                    self.end_h.set(crop_h)

                    self.log_output("Applied pan-right preset")
            except Exception as e:
                self.log_output(f"Error applying preset: {e}")

    def generate_effect(self):
        """Generate Ken Burns effect"""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input image")
            return

        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input image file does not exist")
            return

        # Build command
        cmd = [
            sys.executable,
            "kbe.py",
            "--in",
            self.input_path.get(),
            "--out",
            self.output_path.get(),
            "--estim-path",
            self.estim_path.get(),
            "--refine-path",
            self.refine_path.get(),
            "--inpaint-path",
            self.inpaint_path.get(),
            "--startU",
            str(self.start_u.get()),
            "--startV",
            str(self.start_v.get()),
            "--endU",
            str(self.end_u.get()),
            "--endV",
            str(self.end_v.get()),
            "--startW",
            str(self.start_w.get()),
            "--startH",
            str(self.start_h.get()),
            "--endW",
            str(self.end_w.get()),
            "--endH",
            str(self.end_h.get()),
        ]

        if self.write_frames.get():
            cmd.append("--write-frames")
        if self.dolly.get():
            cmd.append("--dolly")
        if self.twod.get():
            cmd.append("--2d")

        self.log_output(f"Executing command: {' '.join(cmd)}")

        # Disable generate button, enable stop button
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set("Generating...")

        # Run command in thread
        self.generation_thread = threading.Thread(
            target=self.run_generation, args=(cmd,)
        )
        self.generation_thread.start()

    def run_generation(self, cmd):
        """Run the generation command in a separate thread"""
        try:
            # Set CUDA device if available
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
            )

            # Read output line by line
            for line in iter(self.current_process.stdout.readline, ""):
                if line:
                    self.root.after(0, self.log_output, line.strip())

            self.current_process.wait()

            if self.current_process.returncode == 0:
                self.root.after(0, self.generation_completed)
            else:
                self.root.after(0, self.generation_failed)

        except Exception as e:
            self.root.after(0, self.log_output, f"Error: {e}")
            self.root.after(0, self.generation_failed)

    def generation_completed(self):
        """Called when generation completes successfully"""
        self.log_output("Generation completed successfully!")
        self.progress_var.set("Completed")
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.current_process = None

        # Try to load the generated video
        video_path = os.path.join(self.output_path.get(), "output.mp4")
        if os.path.exists(video_path):
            self.load_video_file(video_path)
        else:
            # Look for any mp4 files in the output directory
            output_dir = self.output_path.get()
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(output_dir, file)
                        self.load_video_file(video_path)
                        break

    def generation_failed(self):
        """Called when generation fails"""
        self.log_output("Generation failed!")
        self.progress_var.set("Failed")
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.current_process = None

    def stop_generation(self):
        """Stop the current generation process"""
        if self.current_process:
            self.current_process.terminate()
            self.log_output("Generation stopped by user")
            self.progress_var.set("Stopped")
            self.generate_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.current_process = None

    def load_video(self):
        """Load a video file for preview"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.load_video_file(filename)

    def load_video_file(self, filename):
        """Load video file for playback"""
        try:
            # Close existing video if any
            if self.video_cap:
                self.video_cap.release()
                
            # Stop current playback
            self.stop_video()
            
            # Open new video
            self.video_cap = cv2.VideoCapture(filename)
            
            if not self.video_cap.isOpened():
                self.log_output(f"Error: Could not open video {filename}")
                return
                
            # Get video properties
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            
            # Update progress bar
            self.video_progress.configure(to=self.total_frames - 1)
            self.video_progress.set(0)
            
            # Display first frame
            self.display_current_frame()
            
            # Update time display
            self.update_time_display()
            
            self.log_output(f"Loaded video: {os.path.basename(filename)} ({self.total_frames} frames, {self.video_fps:.1f} fps)")
            
        except Exception as e:
            self.log_output(f"Error loading video: {e}")
            
    def display_current_frame(self):
        """Display the current frame on the canvas"""
        if not self.video_cap:
            return
            
        try:
            # Set frame position
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    h, w = frame.shape[:2]
                    aspect = w / h

                    if canvas_width / canvas_height > aspect:
                        new_height = canvas_height
                        new_width = int(new_height * aspect)
                    else:
                        new_width = canvas_width
                        new_height = int(new_width / aspect)

                    frame = cv2.resize(frame, (new_width, new_height))

                    # Convert to PhotoImage
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image)

                    # Display on canvas
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=photo,
                        anchor=tk.CENTER,
                    )
                    self.video_canvas.image = photo  # Keep reference
                    
        except Exception as e:
            self.log_output(f"Error displaying frame: {e}")

    def toggle_video(self):
        """Toggle video playback"""
        if not self.video_cap:
            self.log_output("No video loaded")
            return
            
        if self.video_playing:
            self.pause_video()
        else:
            self.play_video()
            
    def play_video(self):
        """Start video playback"""
        if not self.video_cap:
            return
            
        self.video_playing = True
        self.video_paused = False
        self.play_pause_btn.config(text="Pause")
        self.play_next_frame()
        
    def pause_video(self):
        """Pause video playback"""
        self.video_playing = False
        self.video_paused = True
        self.play_pause_btn.config(text="Play")
        
        # Cancel scheduled frame update
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
            
    def stop_video(self):
        """Stop video playback and reset to beginning"""
        self.video_playing = False
        self.video_paused = False
        self.play_pause_btn.config(text="Play")
        
        # Cancel scheduled frame update
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
            
        # Reset to first frame
        if self.video_cap:
            self.current_frame = 0
            self.video_progress.set(0)
            self.display_current_frame()
            self.update_time_display()
            
    def play_next_frame(self):
        """Play the next frame"""
        if not self.video_playing or not self.video_cap:
            return
            
        # Check if we've reached the end
        if self.current_frame >= self.total_frames - 1:
            self.stop_video()
            return
            
        # Advance frame and display
        self.current_frame += 1
        self.display_current_frame()
        self.video_progress.set(self.current_frame)
        self.update_time_display()
        
        # Schedule next frame
        delay = int(1000 / self.video_fps)  # Convert to milliseconds
        self.video_after_id = self.root.after(delay, self.play_next_frame)
        
    def seek_video(self, value):
        """Seek to specific frame"""
        if not self.video_cap:
            return
            
        self.current_frame = int(float(value))
        self.display_current_frame()
        self.update_time_display()
        
    def update_time_display(self):
        """Update the time display"""
        if not self.video_cap:
            self.time_label.config(text="00:00 / 00:00")
            return
            
        current_seconds = self.current_frame / self.video_fps
        total_seconds = self.total_frames / self.video_fps
        
        current_time = f"{int(current_seconds // 60):02d}:{int(current_seconds % 60):02d}"
        total_time = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
        
        self.time_label.config(text=f"{current_time} / {total_time}")

    def log_output(self, message):
        """Add message to output log"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    app = KenBurnsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
