from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


SUPPORTED_EXTENSIONS = {
	".png",
	".jpg",
	".jpeg",
	".bmp",
	".gif",
	".webp",
	".tiff",
	".tif",
	".jfif",
	".heic",
	".heif",
	".avif",
}


class FigureDisplayApp:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Photo Viewer")
		self.root.geometry("1000x700")

		self.current_folder: Path | None = None
		self.image_paths: list[Path] = []

		self.original_image: Image.Image | None = None
		self.tk_image: ImageTk.PhotoImage | None = None
		self.image_item: int | None = None

		self.image_scale = 1.0
		self.image_x = 0.0
		self.image_y = 0.0

		self.drag_start_x = 0
		self.drag_start_y = 0
		self.drag_mode: str | None = None

		self.resize_start_y = 0
		self.resize_start_scale = 1.0

		self._build_selector_ui()
		self._bind_shortcuts()

	def _build_selector_ui(self) -> None:
		self.selector_frame = tk.Frame(self.root, bg="#f4f4f4")
		self.selector_frame.pack(fill="both", expand=True)

		title = tk.Label(
			self.selector_frame,
			text="選擇資料夾中的照片",
			font=("Microsoft JhengHei", 20, "bold"),
			bg="#f4f4f4",
		)
		title.pack(pady=(40, 20))

		self.folder_label = tk.Label(
			self.selector_frame,
			text="尚未選擇資料夾",
			font=("Microsoft JhengHei", 11),
			bg="#f4f4f4",
			fg="#333333",
		)
		self.folder_label.pack(pady=(0, 12))

		open_folder_button = tk.Button(
			self.selector_frame,
			text="選擇資料夾",
			font=("Microsoft JhengHei", 11, "bold"),
			padx=18,
			pady=8,
			command=self.choose_folder,
		)
		open_folder_button.pack(pady=(0, 16))

		list_frame = tk.Frame(self.selector_frame)
		list_frame.pack(fill="both", expand=True, padx=120, pady=(0, 40))

		scrollbar = tk.Scrollbar(list_frame)
		scrollbar.pack(side="right", fill="y")

		self.file_listbox = tk.Listbox(
			list_frame,
			font=("Microsoft JhengHei", 11),
			activestyle="none",
			yscrollcommand=scrollbar.set,
		)
		self.file_listbox.pack(side="left", fill="both", expand=True)
		self.file_listbox.bind("<Double-Button-1>", self.open_selected_image)

		scrollbar.config(command=self.file_listbox.yview)

		hint = tk.Label(
			self.selector_frame,
			text="提示：雙擊檔名開啟照片。進入檢視後，左鍵拖曳移動，按住 Ctrl 並拖曳可縮放，Esc 返回清單。",
			font=("Microsoft JhengHei", 10),
			bg="#f4f4f4",
			fg="#555555",
		)
		hint.pack(pady=(0, 18))

		self.viewer_canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, bd=0)
		self.viewer_canvas.bind("<ButtonPress-1>", self.on_left_press)
		self.viewer_canvas.bind("<B1-Motion>", self.on_left_drag)
		self.viewer_canvas.bind("<Configure>", self.on_canvas_resize)

	def _bind_shortcuts(self) -> None:
		self.root.bind("<Escape>", self.back_to_selector)
		self.root.bind("<MouseWheel>", self.on_mouse_wheel)

	def choose_folder(self) -> None:
		selected = filedialog.askdirectory(title="選擇照片資料夾")
		if not selected:
			return

		folder = Path(selected)
		try:
			all_files = sorted(p for p in folder.rglob("*") if p.is_file())
		except Exception as exc:
			messagebox.showerror("讀取失敗", f"無法讀取資料夾：\n{exc}")
			return

		image_files = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTENSIONS]

		self.current_folder = folder
		self.image_paths = image_files

		self.folder_label.config(text=f"目前資料夾：{folder}")
		self.file_listbox.delete(0, tk.END)

		if not image_files:
			ext_summary = ", ".join(sorted({p.suffix.lower() or "(無副檔名)" for p in all_files})[:8])
			if not ext_summary:
				ext_summary = "找不到任何檔案"
			messagebox.showinfo(
				"沒有圖片",
				"此資料夾與其子資料夾內沒有可用的照片格式。\n"
				f"偵測到的副檔名：{ext_summary}",
			)
			return

		for p in image_files:
			self.file_listbox.insert(tk.END, str(p.relative_to(folder)))

	def open_selected_image(self, _event: tk.Event | None = None) -> None:
		selected_indices = self.file_listbox.curselection()
		if not selected_indices:
			return

		idx = selected_indices[0]
		image_path = self.image_paths[idx]
		self.load_image(image_path)

	def load_image(self, image_path: Path) -> None:
		try:
			self.original_image = Image.open(image_path).convert("RGB")
		except Exception as exc:
			messagebox.showerror("讀取失敗", f"無法開啟圖片：\n{exc}")
			return

		self.selector_frame.pack_forget()
		self.viewer_canvas.pack(fill="both", expand=True)
		self.viewer_canvas.focus_set()

		self.root.update_idletasks()
		cw = max(self.viewer_canvas.winfo_width(), 1)
		ch = max(self.viewer_canvas.winfo_height(), 1)

		iw, ih = self.original_image.size
		fit_scale = min(cw / iw, ch / ih)
		self.image_scale = max(fit_scale, 0.05)

		self.image_x = cw / 2
		self.image_y = ch / 2
		self.redraw_image()

	def redraw_image(self) -> None:
		if self.original_image is None:
			return

		iw, ih = self.original_image.size
		target_w = max(int(iw * self.image_scale), 1)
		target_h = max(int(ih * self.image_scale), 1)

		resized = self.original_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
		self.tk_image = ImageTk.PhotoImage(resized)

		self.viewer_canvas.delete("all")
		self.image_item = self.viewer_canvas.create_image(self.image_x, self.image_y, image=self.tk_image)

	def back_to_selector(self, _event: tk.Event | None = None) -> None:
		if not self.viewer_canvas.winfo_ismapped():
			return

		self.viewer_canvas.pack_forget()
		self.selector_frame.pack(fill="both", expand=True)

	def on_left_press(self, event: tk.Event) -> None:
		self.drag_start_x = event.x
		self.drag_start_y = event.y

		if event.state & 0x0004:
			self.drag_mode = "resize"
			self.resize_start_y = event.y
			self.resize_start_scale = self.image_scale
		else:
			self.drag_mode = "move"

	def on_left_drag(self, event: tk.Event) -> None:
		if self.original_image is None:
			return

		if self.drag_mode == "move":
			dx = event.x - self.drag_start_x
			dy = event.y - self.drag_start_y
			self.image_x += dx
			self.image_y += dy
			self.drag_start_x = event.x
			self.drag_start_y = event.y
			self.redraw_image()
			return

		if self.drag_mode == "resize":
			delta = self.resize_start_y - event.y
			factor = 1 + (delta / 300)
			new_scale = self.resize_start_scale * factor
			self.image_scale = max(0.05, min(new_scale, 20.0))
			self.redraw_image()

	def on_mouse_wheel(self, event: tk.Event) -> None:
		if self.original_image is None or not self.viewer_canvas.winfo_ismapped():
			return

		scale_factor = 1.1 if event.delta > 0 else 0.9
		self.image_scale = max(0.05, min(self.image_scale * scale_factor, 20.0))
		self.redraw_image()

	def on_canvas_resize(self, _event: tk.Event) -> None:
		if self.original_image is not None and self.viewer_canvas.winfo_ismapped():
			self.redraw_image()


def main() -> None:
	root = tk.Tk()
	app = FigureDisplayApp(root)
	_ = app
	root.mainloop()


if __name__ == "__main__":
	main()
