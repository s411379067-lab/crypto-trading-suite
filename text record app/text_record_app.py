import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


class IntradayTypingApp:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Typing Launcher")
		self.root.geometry("360x430")
		self.root.resizable(False, False)

		self.typing_window: tk.Toplevel | None = None
		self.copy_preview_window: tk.Toplevel | None = None
		self.shell_frame: tk.Frame | None = None
		self.drag_bar: tk.Frame | None = None
		self.text_widget: tk.Text | None = None
		self.copy_preview_text: tk.Text | None = None
		self.preview_widget: tk.Text | None = None
		self._drag_offset_x = 0
		self._drag_offset_y = 0
		self._resize_start_x = 0
		self._resize_start_y = 0
		self._resize_start_width = 0
		self._resize_start_height = 0
		self.opacity_var = tk.DoubleVar(value=0.78)
		self.opacity_text_var = tk.StringVar(value="78%")
		self.font_size_options = [10, 12, 14, 16, 18, 22, 26, 30]
		self.preset_colors = ["#ffffff", "#089981", "#b22833", "#3179f5", "#ffcc80"]
		self._tag_counter = 0

		container = ttk.Frame(self.root, padding=18)
		container.pack(fill="both", expand=True)

		title = ttk.Label(container, text="Intraday Typing")
		title.pack(pady=(0, 10))

		open_btn = ttk.Button(container, text="Open Typing Box", command=self.open_typing_box)
		open_btn.pack(ipadx=8, ipady=4)

		opacity_title = ttk.Label(container, text="Opacity")
		opacity_title.pack(pady=(14, 4))

		slider_row = ttk.Frame(container)
		slider_row.pack(fill="x")

		opacity_slider = ttk.Scale(
			slider_row,
			from_=0.35,
			to=1.0,
			orient="horizontal",
			variable=self.opacity_var,
			command=self._on_opacity_change,
		)
		opacity_slider.pack(side="left", fill="x", expand=True)

		opacity_value = ttk.Label(slider_row, textvariable=self.opacity_text_var, width=4)
		opacity_value.pack(side="right", padx=(8, 0))

		hint = ttk.Label(container, text="Close with X (with confirmation)")
		hint.pack(pady=(10, 0))

		open_copy_preview_btn = ttk.Button(container, text="Open Copy Preview", command=self.open_copy_preview_box)
		open_copy_preview_btn.pack(fill="x", pady=(10, 0))

		preview_title = ttk.Label(container, text="Auto Copy Preview")
		preview_title.pack(anchor="w", pady=(12, 4))

		self.preview_widget = tk.Text(
			container,
			height=10,
			wrap="word",
			bg="#181c27",
			fg="#f0f0f0",
			insertbackground="#ffffff",
			relief="flat",
			borderwidth=1,
			highlightthickness=1,
			highlightbackground="#4a4a4a",
			font=("Consolas", 12),
			padx=10,
			pady=8,
		)
		self.preview_widget.pack(fill="both", expand=True)
		self.preview_widget.configure(state="disabled")

	def open_typing_box(self) -> None:
		if self.typing_window is not None and self.typing_window.winfo_exists():
			self.typing_window.lift()
			self.typing_window.focus_force()
			return

		win = tk.Toplevel(self.root)
		self.typing_window = win

		# Remove the native title bar to hide minimize/maximize/close controls.
		win.overrideredirect(True)
		win.geometry("760x320+300+220")
		win.configure(bg="#181c27")
		win.attributes("-alpha", self.opacity_var.get())
		win.transient(self.root)

		shell = tk.Frame(win, bg="#181c27", highlightthickness=1, highlightbackground="#4a4a4a")
		shell.pack(fill="both", expand=True, padx=1, pady=1)
		self.shell_frame = shell

		drag_bar = tk.Frame(shell, bg="#181c27", height=18, cursor="fleur")
		drag_bar.pack(fill="x")
		self.drag_bar = drag_bar

		color_toolbar = tk.Frame(shell, bg="#181c27", height=28)
		color_toolbar.pack(fill="x")

		for color in self.preset_colors:
			color_dot = tk.Canvas(
				color_toolbar,
				width=14,
				height=14,
				bg="#181c27",
				highlightthickness=0,
				bd=0,
				cursor="hand2",
			)
			color_dot.create_oval(1, 1, 13, 13, fill=color, outline="#3a445c", width=1)
			color_dot.bind("<Button-1>", lambda _event, c=color: self.apply_color_to_selection(c))
			color_dot.pack(side="left", padx=(6, 0), pady=5)

		close_btn = tk.Button(
			color_toolbar,
			text="X",
			command=self.request_close_typing_box,
			bg="#181c27",
			fg="#e6e6e6",
			bd=0,
			activebackground="#242a38",
			activeforeground="#ffffff",
			cursor="hand2",
			font=("Consolas", 9, "bold"),
		)
		close_btn.pack(side="right", padx=6)

		self.text_widget = tk.Text(
			shell,
			wrap="word",
			undo=True,
			bg="#181c27",
			fg="#f0f0f0",
			insertbackground="#ffffff",
			relief="flat",
			borderwidth=0,
			font=("Consolas", 14),
			padx=14,
			pady=12,
		)
		self.text_widget.pack(fill="both", expand=True)

		resize_handle = tk.Frame(shell, bg="#242a38", width=18, height=18, cursor="size_nw_se")
		resize_handle.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)

		drag_bar.bind("<ButtonPress-1>", self._start_move)
		drag_bar.bind("<B1-Motion>", self._on_move)
		resize_handle.bind("<ButtonPress-1>", self._start_resize)
		resize_handle.bind("<B1-Motion>", self._on_resize)

		win.bind("<FocusIn>", lambda _event: self.text_widget.focus_set() if self.text_widget else None)
		self.text_widget.bind("<Control-Key-1>", lambda event: self._apply_shortcut_color(0, event))
		self.text_widget.bind("<Control-Key-2>", lambda event: self._apply_shortcut_color(1, event))
		self.text_widget.bind("<Control-Key-3>", lambda event: self._apply_shortcut_color(2, event))
		self.text_widget.bind("<Control-Key-4>", lambda event: self._apply_shortcut_color(3, event))
		self.text_widget.bind("<Control-Key-5>", lambda event: self._apply_shortcut_color(4, event))
		self.text_widget.bind("<Control-Key-6>", lambda event: self._open_size_menu_for_widget(self.text_widget, event))
		self.text_widget.bind("<<Modified>>", self._on_text_modified)

		self.text_widget.focus_set()
		self._on_opacity_change("0")
		self._sync_preview_from_typing()

	def request_close_typing_box(self) -> None:
		if self.typing_window is None or not self.typing_window.winfo_exists():
			return

		confirm = messagebox.askyesno("Confirm", "Close typing box?")
		if confirm:
			self.close_typing_box()

	def close_typing_box(self) -> None:
		if self.typing_window is not None and self.typing_window.winfo_exists():
			self.typing_window.destroy()
		self.typing_window = None
		self.shell_frame = None
		self.drag_bar = None
		self.text_widget = None

	def open_copy_preview_box(self) -> None:
		if self.copy_preview_window is not None and self.copy_preview_window.winfo_exists():
			self.copy_preview_window.lift()
			self.copy_preview_window.focus_force()
			if self.preview_widget is not None and self.copy_preview_text is not None:
				self._copy_text_with_tags(self.preview_widget, self.copy_preview_text)
			return

		win = tk.Toplevel(self.root)
		self.copy_preview_window = win

		win.overrideredirect(True)
		win.geometry("760x320+340+260")
		win.configure(bg="#181c27")
		win.attributes("-alpha", self.opacity_var.get())
		win.transient(self.root)

		shell = tk.Frame(win, bg="#181c27", highlightthickness=1, highlightbackground="#4a4a4a")
		shell.pack(fill="both", expand=True, padx=1, pady=1)

		drag_bar = tk.Frame(shell, bg="#181c27", height=18, cursor="fleur")
		drag_bar.pack(fill="x")

		color_toolbar = tk.Frame(shell, bg="#181c27", height=28)
		color_toolbar.pack(fill="x")

		for color in self.preset_colors:
			color_dot = tk.Canvas(
				color_toolbar,
				width=14,
				height=14,
				bg="#181c27",
				highlightthickness=0,
				bd=0,
				cursor="hand2",
			)
			color_dot.create_oval(1, 1, 13, 13, fill=color, outline="#3a445c", width=1)
			color_dot.bind("<Button-1>", lambda _event, c=color: self.apply_color_to_copy_selection(c))
			color_dot.pack(side="left", padx=(6, 0), pady=5)

		close_btn = tk.Button(
			color_toolbar,
			text="X",
			command=self.request_close_copy_preview_box,
			bg="#181c27",
			fg="#e6e6e6",
			bd=0,
			activebackground="#242a38",
			activeforeground="#ffffff",
			cursor="hand2",
			font=("Consolas", 9, "bold"),
		)
		close_btn.pack(side="right", padx=6)

		self.copy_preview_text = tk.Text(
			shell,
			wrap="word",
			undo=True,
			bg="#181c27",
			fg="#f0f0f0",
			insertbackground="#ffffff",
			relief="flat",
			borderwidth=0,
			font=("Consolas", 14),
			padx=14,
			pady=12,
		)
		self.copy_preview_text.pack(fill="both", expand=True)
		self.copy_preview_text.bind("<Control-Key-1>", lambda event: self._apply_copy_shortcut_color(0, event))
		self.copy_preview_text.bind("<Control-Key-2>", lambda event: self._apply_copy_shortcut_color(1, event))
		self.copy_preview_text.bind("<Control-Key-3>", lambda event: self._apply_copy_shortcut_color(2, event))
		self.copy_preview_text.bind("<Control-Key-4>", lambda event: self._apply_copy_shortcut_color(3, event))
		self.copy_preview_text.bind("<Control-Key-5>", lambda event: self._apply_copy_shortcut_color(4, event))
		self.copy_preview_text.bind("<Control-Key-6>", lambda event: self._open_size_menu_for_widget(self.copy_preview_text, event))

		resize_handle = tk.Frame(shell, bg="#242a38", width=18, height=18, cursor="size_nw_se")
		resize_handle.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)

		if self.preview_widget is not None:
			self._copy_text_with_tags(self.preview_widget, self.copy_preview_text)

		move_start_x = 0
		move_start_y = 0
		win_start_x = 0
		win_start_y = 0
		resize_start_x = 0
		resize_start_y = 0
		resize_start_width = 0
		resize_start_height = 0

		def on_start_move(event: tk.Event) -> None:
			nonlocal move_start_x, move_start_y, win_start_x, win_start_y
			move_start_x = event.x_root
			move_start_y = event.y_root
			win_start_x = win.winfo_x()
			win_start_y = win.winfo_y()

		def on_move(event: tk.Event) -> None:
			delta_x = event.x_root - move_start_x
			delta_y = event.y_root - move_start_y
			win.geometry(f"+{win_start_x + delta_x}+{win_start_y + delta_y}")

		def on_start_resize(event: tk.Event) -> None:
			nonlocal resize_start_x, resize_start_y, resize_start_width, resize_start_height
			resize_start_x = event.x_root
			resize_start_y = event.y_root
			resize_start_width = win.winfo_width()
			resize_start_height = win.winfo_height()

		def on_resize(event: tk.Event) -> None:
			delta_x = event.x_root - resize_start_x
			delta_y = event.y_root - resize_start_y
			new_width = resize_start_width + delta_x
			new_height = resize_start_height + delta_y
			current_x = win.winfo_x()
			current_y = win.winfo_y()
			win.geometry(f"{new_width}x{new_height}+{current_x}+{current_y}")

		drag_bar.bind("<ButtonPress-1>", on_start_move)
		drag_bar.bind("<B1-Motion>", on_move)
		resize_handle.bind("<ButtonPress-1>", on_start_resize)
		resize_handle.bind("<B1-Motion>", on_resize)

	def request_close_copy_preview_box(self) -> None:
		if self.copy_preview_window is None or not self.copy_preview_window.winfo_exists():
			return

		confirm = messagebox.askyesno("Confirm", "Close copy preview box?")
		if confirm:
			self.close_copy_preview_box()

	def close_copy_preview_box(self) -> None:
		if self.copy_preview_window is not None and self.copy_preview_window.winfo_exists():
			self.copy_preview_window.destroy()
		self.copy_preview_window = None
		self.copy_preview_text = None

	def _start_move(self, event: tk.Event) -> None:
		self._drag_offset_x = event.x_root
		self._drag_offset_y = event.y_root

	def _start_resize(self, event: tk.Event) -> None:
		if not self.typing_window or not self.typing_window.winfo_exists():
			return

		self._resize_start_x = event.x_root
		self._resize_start_y = event.y_root
		self._resize_start_width = self.typing_window.winfo_width()
		self._resize_start_height = self.typing_window.winfo_height()

	def _on_move(self, event: tk.Event) -> None:
		if not self.typing_window or not self.typing_window.winfo_exists():
			return

		delta_x = event.x_root - self._drag_offset_x
		delta_y = event.y_root - self._drag_offset_y

		current_x = self.typing_window.winfo_x()
		current_y = self.typing_window.winfo_y()

		self.typing_window.geometry(f"+{current_x + delta_x}+{current_y + delta_y}")
		self._drag_offset_x = event.x_root
		self._drag_offset_y = event.y_root

	def _on_resize(self, event: tk.Event) -> None:
		if not self.typing_window or not self.typing_window.winfo_exists():
			return

		delta_x = event.x_root - self._resize_start_x
		delta_y = event.y_root - self._resize_start_y

		new_width = self._resize_start_width + delta_x
		new_height = self._resize_start_height + delta_y

		current_x = self.typing_window.winfo_x()
		current_y = self.typing_window.winfo_y()
		self.typing_window.geometry(f"{new_width}x{new_height}+{current_x}+{current_y}")

	def _on_opacity_change(self, _value: str) -> None:
		opacity = max(0.35, min(1.0, self.opacity_var.get()))
		self.opacity_text_var.set(f"{int(round(opacity * 100))}%")
		if self.typing_window is not None and self.typing_window.winfo_exists():
			self.typing_window.attributes("-alpha", opacity)
		if self.copy_preview_window is not None and self.copy_preview_window.winfo_exists():
			self.copy_preview_window.attributes("-alpha", opacity)

	def apply_color_to_selection(self, color: str) -> None:
		if self.text_widget is None or not self.text_widget.winfo_exists():
			return

		try:
			start = self.text_widget.index("sel.first")
			end = self.text_widget.index("sel.last")
		except tk.TclError:
			return

		tag_name = f"fg_{self._tag_counter}"
		self._tag_counter += 1
		self.text_widget.tag_configure(tag_name, foreground=color)
		self.text_widget.tag_add(tag_name, start, end)
		self._sync_preview_from_typing()

	def _on_text_modified(self, _event: tk.Event | None = None) -> None:
		if self.text_widget is None or not self.text_widget.winfo_exists():
			return

		if self.text_widget.edit_modified():
			self._sync_preview_from_typing()
			self.text_widget.edit_modified(False)

	def _sync_preview_from_typing(self) -> None:
		if self.text_widget is None or not self.text_widget.winfo_exists():
			return
		if self.preview_widget is None or not self.preview_widget.winfo_exists():
			return

		content = self.text_widget.get("1.0", "end-1c")

		self.preview_widget.configure(state="normal")
		self.preview_widget.delete("1.0", "end")
		for tag in self.preview_widget.tag_names():
			if tag != "sel":
				self.preview_widget.tag_delete(tag)

		if content:
			self.preview_widget.insert("1.0", content)

		for tag in self.text_widget.tag_names():
			if tag == "sel":
				continue

			tag_cfg = self.text_widget.tag_configure(tag)
			tag_options: dict[str, str] = {}
			for option in ("foreground", "font", "background"):
				value_tuple = tag_cfg.get(option)
				if isinstance(value_tuple, tuple) and len(value_tuple) >= 5 and value_tuple[4]:
					tag_options[option] = value_tuple[4]

			if tag_options:
				self.preview_widget.tag_configure(tag, **tag_options)

			ranges = self.text_widget.tag_ranges(tag)
			for i in range(0, len(ranges), 2):
				self.preview_widget.tag_add(tag, ranges[i], ranges[i + 1])

		self.preview_widget.configure(state="disabled")

		if self.copy_preview_text is not None and self.copy_preview_text.winfo_exists():
			self._copy_text_with_tags(self.preview_widget, self.copy_preview_text)

	def paste_preview_to_typing(self) -> None:
		if self.preview_widget is None or not self.preview_widget.winfo_exists():
			return

		if self.text_widget is None or not self.text_widget.winfo_exists():
			self.open_typing_box()

		if self.text_widget is None or not self.text_widget.winfo_exists():
			return

		self._copy_text_with_tags(self.preview_widget, self.text_widget)
		self.text_widget.focus_set()

	def _copy_text_with_tags(self, source: tk.Text, target: tk.Text) -> None:
		content = source.get("1.0", "end-1c")

		target.delete("1.0", "end")
		for tag in target.tag_names():
			if tag != "sel":
				target.tag_delete(tag)

		if content:
			target.insert("1.0", content)

		for tag in source.tag_names():
			if tag == "sel":
				continue

			tag_cfg = source.tag_configure(tag)
			tag_options: dict[str, str] = {}
			for option in ("foreground", "font", "background"):
				value_tuple = tag_cfg.get(option)
				if isinstance(value_tuple, tuple) and len(value_tuple) >= 5 and value_tuple[4]:
					tag_options[option] = value_tuple[4]

			if tag_options:
				target.tag_configure(tag, **tag_options)

			ranges = source.tag_ranges(tag)
			for i in range(0, len(ranges), 2):
				target.tag_add(tag, ranges[i], ranges[i + 1])

	def _apply_shortcut_color(self, index: int, _event: tk.Event | None = None) -> str:
		if index < 0 or index >= len(self.preset_colors):
			return "break"

		self.apply_color_to_selection(self.preset_colors[index])
		return "break"

	def apply_color_to_copy_selection(self, color: str) -> None:
		if self.copy_preview_text is None or not self.copy_preview_text.winfo_exists():
			return

		try:
			start = self.copy_preview_text.index("sel.first")
			end = self.copy_preview_text.index("sel.last")
		except tk.TclError:
			return

		tag_name = f"copy_fg_{self._tag_counter}"
		self._tag_counter += 1
		self.copy_preview_text.tag_configure(tag_name, foreground=color)
		self.copy_preview_text.tag_add(tag_name, start, end)

	def _apply_copy_shortcut_color(self, index: int, _event: tk.Event | None = None) -> str:
		if index < 0 or index >= len(self.preset_colors):
			return "break"

		self.apply_color_to_copy_selection(self.preset_colors[index])
		return "break"

	def _open_size_menu_for_widget(self, widget: tk.Text | None, _event: tk.Event | None = None) -> str:
		if widget is None or not widget.winfo_exists():
			return "break"

		try:
			start = widget.index("sel.first")
			end = widget.index("sel.last")
		except tk.TclError:
			return "break"

		menu = tk.Menu(widget, tearoff=0, bg="#1f2432", fg="#f0f0f0", activebackground="#2d3447", activeforeground="#ffffff")
		for size in self.font_size_options:
			menu.add_command(
				label=str(size),
				command=lambda s=size, st=start, ed=end, w=widget: self._apply_size_to_widget_range(w, s, st, ed),
			)

		x_pos = widget.winfo_pointerx()
		y_pos = widget.winfo_pointery()
		menu.tk_popup(x_pos, y_pos)
		menu.grab_release()
		return "break"

	def _apply_size_to_widget_range(self, widget: tk.Text, size: int, start: str, end: str) -> None:
		if not widget.winfo_exists():
			return

		tag_name = f"size_{self._tag_counter}"
		self._tag_counter += 1
		widget.tag_configure(tag_name, font=("Consolas", size))
		widget.tag_add(tag_name, start, end)

		if self.text_widget is not None and widget == self.text_widget:
			self._sync_preview_from_typing()


def main() -> None:
	root = tk.Tk()
	app = IntradayTypingApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
