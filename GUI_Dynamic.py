import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import cv2
import torch
from PIL import Image, ImageTk, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
from segmentation_models_pytorch import DeepLabV3Plus
import threading


###########################################
# Initialize Diffusers Inpainting Pipeline
###########################################
print("Loading Stable Diffusion Inpainting Pipeline...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


###########################################
# Load Hair Segmentation Model
###########################################
def load_segmentation_model(model_path, device):
    model = DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


print("Loading Hair Segmentation Model...")
model = load_segmentation_model("best_hair_seg_model.pth", device)


###########################################
# Hair Segmentation Functions
###########################################
def preprocess_pil(pil_img, size=(256, 256)):
    arr = np.array(pil_img.resize(size), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # Convert to [C, H, W]
    return torch.from_numpy(arr).unsqueeze(0)  # Add batch dimension


def segment_hair(model, pil_img):
    t = preprocess_pil(pil_img, size=(256, 256)).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.sigmoid(logits)[0, 0]  # Take the hair channel
    mask_small = (probs.cpu().numpy() > 0.5).astype(np.uint8)  # Binarize
    return mask_small


def upsample_mask(mask_small, out_size):
    w, h = out_size
    up = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return up.astype(np.uint8)


def draw_circle(mask_img, x, y, radius, value):
    h, w = mask_img.shape
    cv2.circle(mask_img, (x, y), radius, (value,), -1)


def overlay_mask_red(pil_img, mask_full):
    """
    Overlay the mask on the image in red.
    """
    base_np = np.array(pil_img, dtype=np.uint8)
    mask_np = mask_full.astype(np.float32)[..., None]
    alpha = 0.4
    red_overlay = np.array([255, 0, 0], dtype=np.float32)
    overlaid = base_np * (1 - alpha * mask_np) + red_overlay * (alpha * mask_np)
    return Image.fromarray(np.clip(overlaid, 0, 255).astype(np.uint8))


###########################################
# Generate Realistic Hair with Diffusers
###########################################
def generate_realistic_hair(
    original_img: Image.Image,
    mask_full: np.ndarray,
    prompt: str,
    negative_prompt: str,
    steps: int = 30,
    guidance_scale: float = 7.5,
) -> Image.Image:
    w, h = original_img.size
    if mask_full.shape != (h, w):
        print("Resizing mask to match image dimensions.")
        mask_full = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_full = cv2.GaussianBlur(mask_full, (5, 5), 0).astype(np.uint8)
    mask_pil = Image.fromarray((mask_full * 255).astype(np.uint8)).convert("L")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_img,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]

    result = result.filter(ImageFilter.SHARPEN)
    return result


###########################################
# Utility: Resize Image If Needed
###########################################
def resize_image_if_needed(pil_img, max_size=1024):
    """
    Resizes image to fit within max_size x max_size, preserving aspect ratio.
    """
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img


###########################################
# Tkinter GUI for Hair Editing
###########################################
class HairApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Hair Editor")
        self.geometry("1400x800")

        # --- Black Theme Colors ---
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.btn_bg = "#3a3a3a"
        self.canvas_bg = "#2f2f2f"

        self.configure(bg=self.bg_color)

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab for Generative Edit
        self.tab_gen = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.tab_gen, text="Generative Edit")

        # Variables
        self.original_pil = None
        self.mask_full = None
        self.result_pil_gen = None
        self.brush_radius = 10
        self.paint_value = 1
        self.edit_mask_mode = False
        self.color_hex = None

        self.create_tab_gen()

    ##############################
    # TAB 1: Generative Edit
    ##############################
    def create_tab_gen(self):
        top_frame = tk.Frame(self.tab_gen, bg=self.bg_color)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Button(
            top_frame, text="Open Image", command=self.open_image_gen,
            bg=self.btn_bg, fg=self.fg_color
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top_frame, text="Segment Hair", command=self.segment_hair_gen,
            bg=self.btn_bg, fg=self.fg_color
        ).pack(side=tk.LEFT, padx=5)

        self.edit_mask_btn = tk.Button(
            top_frame, text="Edit Mask", command=self.toggle_edit_mask,
            bg=self.btn_bg, fg=self.fg_color
        )
        self.edit_mask_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Brush Radius:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(top_frame, from_=1, to=50, value=10, command=self.on_brush_slider)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.brush_label = tk.Label(top_frame, text="10", bg=self.bg_color, fg=self.fg_color)
        self.brush_label.pack(side=tk.LEFT)

        # Paint/Erase radio buttons
        paint_radio_var = tk.IntVar(value=1)
        self.paint_radio = tk.Radiobutton(
            top_frame, text="Paint", variable=paint_radio_var, value=1,
            command=lambda: self.set_paint_mode(1), bg=self.bg_color,
            fg=self.fg_color, selectcolor=self.btn_bg
        )
        self.paint_radio.pack(side=tk.LEFT, padx=5)

        self.erase_radio = tk.Radiobutton(
            top_frame, text="Erase", variable=paint_radio_var, value=0,
            command=lambda: self.set_paint_mode(0), bg=self.bg_color,
            fg=self.fg_color, selectcolor=self.btn_bg
        )
        self.erase_radio.pack(side=tk.LEFT, padx=5)

        # Prompt and color
        prompt_frame = tk.Frame(self.tab_gen, bg=self.bg_color)
        prompt_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        tk.Label(prompt_frame, text="Prompt:", bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar(value="A person with beautiful hair")
        self.prompt_entry = tk.Entry(prompt_frame, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(
            prompt_frame, text="Pick Color", command=self.pick_color,
            bg=self.btn_bg, fg=self.fg_color
        ).pack(side=tk.LEFT, padx=5)

        self.color_preview = tk.Canvas(prompt_frame, width=20, height=20, bg="white")
        self.color_preview.pack(side=tk.LEFT, padx=5)

        tk.Button(
            prompt_frame, text="Generate", command=self.start_generation,
            bg=self.btn_bg, fg=self.fg_color
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            prompt_frame, text="Save", command=self.save_result_gen,
            bg=self.btn_bg, fg=self.fg_color
        ).pack(side=tk.LEFT, padx=5)

        # Container for canvases
        container = tk.Frame(self.tab_gen, bg=self.bg_color)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_gen = tk.Canvas(container, bg=self.canvas_bg, width=800, height=600)
        self.canvas_gen.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_mask = tk.Canvas(container, bg=self.canvas_bg, width=400, height=600)
        self.canvas_mask.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mask editing events
        self.canvas_gen.bind("<Button-1>", self.on_mask_click)
        self.canvas_gen.bind("<B1-Motion>", self.on_mask_click)

    def toggle_edit_mask(self):
        self.edit_mask_mode = not self.edit_mask_mode
        self.edit_mask_btn.config(text="Editing Mask" if self.edit_mask_mode else "Edit Mask")
        self.update_canvas_gen()

    def pick_color(self):
        color_info = colorchooser.askcolor(title="Pick Hair Color")
        if color_info and color_info[1]:
            self.color_hex = color_info[1]
            self.color_preview.config(bg=self.color_hex)
            messagebox.showinfo("Color", f"Picked color: {self.color_hex}")

    def start_generation(self):
        if not self.original_pil or self.mask_full is None:
            messagebox.showwarning("Warning", "No image or mask available!")
            return
        threading.Thread(target=self.generate_realistic).start()

    def generate_realistic(self):
        prompt = self.prompt_var.get()
        if self.color_hex:
            prompt += f", {self.color_hex} hair"

        neg_prompt = "mutated, deformed, low quality"
        result_img = generate_realistic_hair(
            self.original_pil, self.mask_full, prompt, neg_prompt
        )
        self.result_pil_gen = result_img
        self.display_image_on_canvas(self.canvas_gen, result_img)

    def on_brush_slider(self, val):
        self.brush_radius = int(float(val))
        self.brush_label.config(text=str(self.brush_radius))

    def set_paint_mode(self, val):
        self.paint_value = val

    def open_image_gen(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            pil_img = Image.open(path).convert("RGB")

            # --- Resize if needed to avoid offset issues on very large images ---
            pil_img = resize_image_if_needed(pil_img, max_size=1024)

            self.original_pil = pil_img
            self.mask_full = None
            self.result_pil_gen = pil_img
            self.update_canvas_gen()

    def segment_hair_gen(self):
        if not self.original_pil:
            messagebox.showwarning("Warning", "No image loaded for segmentation!")
            return
        mask_small = segment_hair(model, self.original_pil)
        self.mask_full = upsample_mask(mask_small, self.original_pil.size)
        self.update_mask_display()

    def on_mask_click(self, event):
        if not self.edit_mask_mode or not self.original_pil or self.mask_full is None:
            return

        canvas_width = self.canvas_gen.winfo_width()
        canvas_height = self.canvas_gen.winfo_height()
        img_width, img_height = self.original_pil.size

        # Calculate the scale and offset for proper coordinate mapping
        scale = min(canvas_width / img_width, canvas_height / img_height)
        offset_x = (canvas_width - scale * img_width) / 2
        offset_y = (canvas_height - scale * img_height) / 2

        x_img = int((event.x - offset_x) / scale)
        y_img = int((event.y - offset_y) / scale)

        if x_img < 0 or y_img < 0 or x_img >= img_width or y_img >= img_height:
            return

        draw_circle(self.mask_full, x_img, y_img, self.brush_radius, self.paint_value)
        self.update_canvas_gen()
        self.update_mask_display()

    def update_mask_display(self):
        if self.mask_full is not None:
            mask_pil = Image.fromarray((self.mask_full * 255).astype(np.uint8)).convert("L")
            self.display_image_on_canvas(self.canvas_mask, mask_pil)

    def update_canvas_gen(self):
        if not self.original_pil:
            return

        if self.edit_mask_mode and self.mask_full is not None:
            overlay = overlay_mask_red(self.original_pil, self.mask_full)
            self.display_image_on_canvas(self.canvas_gen, overlay)
        else:
            self.display_image_on_canvas(self.canvas_gen, self.original_pil)

    def save_result_gen(self):
        if not self.result_pil_gen:
            messagebox.showwarning("Warning", "No result to save!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.result_pil_gen.save(path)

    def display_image_on_canvas(self, canvas, pil_img):
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        img_width, img_height = pil_img.size

        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        pil_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_resized)

        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas.image = tk_img


###########################################
# Main
###########################################
if __name__ == "__main__":
    app = HairApp()
    app.mainloop()
