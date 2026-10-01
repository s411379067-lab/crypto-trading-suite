from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Any


DRAWING_TEMPLATE_VERSION = "1.0"


TYPE_TO_CATEGORY = {
    "horizontal_line": "line",
    "trend_line": "line",
    "rectangle": "rectangle",
    "fibonacci": "fibonacci",
    "text": "text",
}


class DrawingTemplateRepository:
    """Filesystem-backed drawing-style template repository.

    Templates deliberately contain appearance/configuration only. Drawing geometry,
    coordinates and text content remain owned by the Case JSON.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(__file__).resolve().parents[1] / "drawing template"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for category in sorted(set(TYPE_TO_CATEGORY.values())):
            (self.root / category).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def category_for_drawing_type(drawing_type: str) -> str:
        return TYPE_TO_CATEGORY.get(str(drawing_type), str(drawing_type))

    @staticmethod
    def _safe_filename(name: str) -> str:
        text = str(name).strip()
        if not text:
            raise ValueError("模板名稱不可空白")
        # Windows-forbidden filename characters and trailing dots/spaces.
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).rstrip(" .")
        if not text:
            text = "template"
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if text.upper() in reserved:
            text = f"_{text}"
        return text[:120]

    def category_dir(self, category: str) -> Path:
        path = self.root / str(category)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def template_path(self, category: str, name: str) -> Path:
        return self.category_dir(category) / f"{self._safe_filename(name)}.json"

    def exists(self, category: str, name: str) -> bool:
        return self.template_path(category, name).exists()

    def save(self, category: str, name: str, payload: dict[str, Any]) -> Path:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("模板名稱不可空白")
        data = {
            "template_version": DRAWING_TEMPLATE_VERSION,
            "name": clean_name,
            "drawing_type": str(category),
            **deepcopy(payload),
        }
        path = self.template_path(category, clean_name)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        return path

    def load_path(self, path: str | Path) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Drawing template 必須是 JSON object")
        return data

    def list(self, category: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self.category_dir(category).glob("*.json"):
            try:
                data = self.load_path(path)
                if str(data.get("drawing_type", category)) != str(category):
                    continue
                data["_path"] = str(path)
                items.append(data)
            except Exception:
                # A manually edited/broken template must not break the drawing menu.
                continue
        items.sort(key=lambda x: str(x.get("name", "")).casefold())
        return items

    def delete(self, category: str, name: str) -> bool:
        path = self.template_path(category, name)
        if not path.exists():
            return False
        path.unlink()
        return True
