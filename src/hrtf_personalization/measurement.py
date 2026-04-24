from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import cv2
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MeasurementSpec:
    key: str
    label: str
    kind: str
    image: str
    instructions: str
    point_labels: tuple[str, ...] = ()


@dataclass
class CalibratedImage:
    image: np.ndarray
    transform: np.ndarray


@dataclass
class Landmark:
    label: str
    point: np.ndarray


_POINT_PICK_RADIUS_PX = 18.0
_TITLE_WRAP_WIDTH = 72
_STATUS_WRAP_WIDTH = 96


MEASUREMENT_SPECS: list[MeasurementSpec] = [
    MeasurementSpec(
        "x1",
        "Head width",
        "distance",
        "front",
        "Click the farthest left and farthest right visible edges of the head in the front view. Ignore loose hair if possible.",
        ("Head left", "Head right"),
    ),
    MeasurementSpec(
        "x2",
        "Head height",
        "distance",
        "side",
        "Click the highest visible point of the skull and the lowest visible point of the head profile in the side view.",
        ("Head top", "Head bottom"),
    ),
    MeasurementSpec(
        "x3",
        "Head depth",
        "distance",
        "side",
        "Click the most forward point and the most rearward point of the head profile in the side view.",
        ("Head front", "Head back"),
    ),
    MeasurementSpec(
        "x4",
        "Pinna offset down",
        "center_distance",
        "side",
        "Click the ear landmark you want to measure relative to the head center, usually the center of the ear opening or ear canal area. This step represents how far below the head center the pinna sits.",
        ("Ear landmark",),
    ),
    MeasurementSpec(
        "x5",
        "Pinna offset back",
        "center_distance",
        "side",
        "Click the ear landmark you want to measure relative to the head center, usually the center of the ear opening or ear canal area. This step represents how far behind the head center the pinna sits.",
        ("Ear landmark",),
    ),
    MeasurementSpec(
        "x6",
        "Neck width",
        "distance",
        "front",
        "Click the left and right outer edges of the neck where it is widest in the front view.",
        ("Neck left", "Neck right"),
    ),
    MeasurementSpec(
        "x7",
        "Neck height",
        "distance",
        "side",
        "Click the top and bottom of the visible neck region in the side view.",
        ("Neck top", "Neck bottom"),
    ),
    MeasurementSpec(
        "x8",
        "Neck depth",
        "distance",
        "side",
        "Click the most forward and most rearward points of the neck region in the side view.",
        ("Neck front", "Neck back"),
    ),
    MeasurementSpec(
        "x9",
        "Torso top width",
        "distance",
        "front",
        "Click the left and right outer edges of the upper torso in the front view.",
        ("Torso left", "Torso right"),
    ),
    MeasurementSpec(
        "x10",
        "Torso top height",
        "distance",
        "side",
        "Click the top and bottom limits of the upper torso region in the side view.",
        ("Torso top", "Torso bottom"),
    ),
    MeasurementSpec(
        "x11",
        "Torso top depth",
        "distance",
        "side",
        "Click the most forward and most rearward points of the upper torso in the side view.",
        ("Torso front", "Torso back"),
    ),
    MeasurementSpec(
        "x12",
        "Shoulder width",
        "distance",
        "front",
        "Click the left and right outer shoulder points in the front view.",
        ("Shoulder left", "Shoulder right"),
    ),
    MeasurementSpec(
        "x13",
        "Head offset forward",
        "center_distance",
        "side",
        "Click the forward landmark to measure from the head center, usually the most forward point on the forehead or face profile.",
        ("Face front",),
    ),
    MeasurementSpec("x14", "Height in inches", "numeric", "manual", "Type the person's standing height in inches."),
    MeasurementSpec("x15", "Seated height", "numeric", "manual", "Type the person's seated height in inches."),
    MeasurementSpec("x16", "Head circumference", "numeric", "manual", "Type the measured head circumference in inches."),
    MeasurementSpec("x17", "Shoulder circumference", "numeric", "manual", "Type the measured shoulder circumference in inches."),
]


def collect_anthropometrics(
    front_image_path: str | Path,
    side_image_path: str | Path,
    output_path: str | Path | None = None,
) -> np.ndarray:
    front_image = _load_image(front_image_path)
    side_image = _load_image(side_image_path)

    front_calibrated = _prepare_calibrated_image(front_image, "Front image")
    side_calibrated = _prepare_calibrated_image(side_image, "Side image")

    values: list[float] = []
    front_landmarks: list[Landmark] = []
    side_landmarks: list[Landmark] = []
    side_center, side_center_landmark = _capture_center(
        side_calibrated,
        "Side image",
        "Click the approximate center of the head in the side view. Use the midpoint of the visible head profile from top-to-bottom and front-to-back.",
        existing_landmarks=side_landmarks,
    )
    side_landmarks.append(side_center_landmark)

    for spec in MEASUREMENT_SPECS:
        if spec.kind == "distance":
            image = front_calibrated if spec.image == "front" else side_calibrated
            existing_landmarks = front_landmarks if spec.image == "front" else side_landmarks
            value, new_landmarks = _capture_distance(
                image,
                spec.label,
                spec.instructions,
                point_labels=spec.point_labels,
                existing_landmarks=existing_landmarks,
            )
            existing_landmarks.extend(new_landmarks)
        elif spec.kind == "center_distance":
            value, landmark = _capture_center_distance(
                side_calibrated,
                side_center,
                spec.label,
                spec.instructions,
                point_label=spec.point_labels[0] if spec.point_labels else spec.label,
                existing_landmarks=side_landmarks,
            )
            side_landmarks.append(landmark)
        elif spec.kind == "numeric":
            value = _capture_numeric(spec.label, spec.instructions)
        else:
            msg = f"Unsupported measurement kind: {spec.kind}"
            raise ValueError(msg)
        values.append(value)

    anthropometrics = np.asarray(values, dtype=np.float32)
    if output_path is not None:
        _write_measurement_json(output_path, anthropometrics)
    return anthropometrics


def load_anthropometrics_json(path: str | Path) -> np.ndarray:
    payload = Path(path).read_text(encoding="utf-8")
    data = np.asarray(_parse_json_values(payload), dtype=np.float32)
    if data.shape[0] != 17:
        raise ValueError(f"Expected 17 values, found {data.shape[0]}.")
    return data


def _write_measurement_json(output_path: str | Path, anthropometrics: np.ndarray) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "order": [spec.key for spec in MEASUREMENT_SPECS],
        "labels": [spec.label for spec in MEASUREMENT_SPECS],
        "values_cm": anthropometrics.tolist(),
    }
    path.write_text(_dump_json(payload), encoding="utf-8")


def _dump_json(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)


def _parse_json_values(text: str) -> list[float]:
    import json

    payload = json.loads(text)
    if isinstance(payload, dict) and "values_cm" in payload:
        return list(payload["values_cm"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Anthropometric JSON must contain a `values_cm` list or be a raw list.")


def _load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _prepare_calibrated_image(image: np.ndarray, title: str) -> CalibratedImage:
    auto_points = _detect_document_corners(image)
    if auto_points is None or not _confirm_auto_detection(image, auto_points, title):
        auto_points = _click_ordered_corners(image, title)
    return CalibratedImage(image=image, transform=_build_document_transform(auto_points))


def _detect_document_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)
    if best is None or best_area < 10000:
        return None
    return _order_quadrilateral(best.astype(np.float32))


def _confirm_auto_detection(image: np.ndarray, points: np.ndarray, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    overlay = np.vstack([points, points[0]])
    ax.plot(overlay[:, 0], overlay[:, 1], "-r", linewidth=2)
    labels = ["1", "2", "3", "4"]
    for index, point in enumerate(points):
        ax.annotate(labels[index], point, color="yellow", fontsize=14, fontweight="bold")
    _set_wrapped_title(ax, f"{title}: auto-detected calibration card. Close the window or press Enter to accept.")
    ax.axis("off")
    plt.show(block=False)
    response = input("Accept detected page corners? [Y/n]: ").strip().lower()
    plt.close(fig)
    return response in {"", "y", "yes"}


def _click_ordered_corners(image: np.ndarray, title: str) -> np.ndarray:
    labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    points: list[tuple[float, float]] = []
    selected_landmarks: list[Landmark] = []
    for index, label in enumerate(labels, start=1):
        point = _click_points(
            image,
            1,
            f"{title}: click corner {index}/4 - {label}",
            existing_landmarks=selected_landmarks,
            point_labels=(label,),
        )[0]
        points.append((float(point[0]), float(point[1])))
        selected_landmarks.append(Landmark(label, point))
    return np.asarray(points, dtype=np.float32)


def _click_points(
    image: np.ndarray,
    count: int,
    title: str,
    existing_landmarks: list[Landmark] | None = None,
    point_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    session = _PointSelectionSession(
        image=image,
        count=count,
        title=title,
        existing_landmarks=existing_landmarks or [],
        point_labels=_resolve_point_labels(count, point_labels),
    )
    return session.run()


def _document_destination_points() -> np.ndarray:
    width_px = 850
    height_px = 1100
    return np.array(
        [[0.0, 0.0], [width_px - 1.0, 0.0], [width_px - 1.0, height_px - 1.0], [0.0, height_px - 1.0]],
        dtype=np.float32,
    )


def _build_document_transform(points: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(points, _document_destination_points())


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(1, -1, 2), transform).reshape(-1, 2)


def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _draw_landmarks(ax: Axes, landmarks: list[Landmark]) -> None:
    for landmark in landmarks:
        ax.scatter(landmark.point[0], landmark.point[1], s=55, c="#00d4ff", edgecolors="black", linewidths=0.75)
        ax.annotate(
            landmark.label,
            xy=(float(landmark.point[0]), float(landmark.point[1])),
            xytext=(8, 8),
            textcoords="offset points",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "black", "ec": "#00d4ff", "alpha": 0.75},
        )


def _wrap_display_text(text: str, width: int) -> str:
    return fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _set_wrapped_title(ax: Axes, title: str) -> None:
    ax.set_title(_wrap_display_text(title, _TITLE_WRAP_WIDTH), wrap=True, pad=16)


def _resolve_point_labels(count: int, point_labels: tuple[str, ...] | None) -> tuple[str, ...]:
    if point_labels is None:
        return tuple(f"Point {index}" for index in range(1, count + 1))
    if len(point_labels) != count:
        raise ValueError(f"Expected {count} point labels, got {len(point_labels)}.")
    return point_labels


class _PointSelectionSession:
    def __init__(
        self,
        image: np.ndarray,
        count: int,
        title: str,
        existing_landmarks: list[Landmark],
        point_labels: tuple[str, ...],
    ) -> None:
        self.image = image
        self.count = count
        self.title = title
        self.existing_landmarks = existing_landmarks
        self.point_labels = point_labels
        self.points: list[np.ndarray] = []
        self.point_annotations: list = []
        self.active_index: int | None = None
        self.confirmed = False
        self.fig = None
        self.ax = None
        self.current_scatter = None
        self.status_text = None
        self.confirm_button = None

    def run(self) -> np.ndarray:
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.14, top=0.88)
        self.ax.imshow(self.image)
        if self.existing_landmarks:
            _draw_landmarks(self.ax, self.existing_landmarks)
        _set_wrapped_title(self.ax, self.title)
        self.ax.axis("off")
        self.current_scatter = self.ax.scatter([], [], s=75, c="#ffb000", edgecolors="black", linewidths=1.0)
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=10, color="black", wrap=True)
        button_ax = self.fig.add_axes([0.82, 0.02, 0.14, 0.06])
        self.confirm_button = Button(button_ax, "Confirm")
        self.confirm_button.on_clicked(self._on_confirm)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._refresh()
        plt.show(block=True)
        if not self.confirmed:
            raise RuntimeError("Point selection cancelled before confirmation.")
        return np.asarray(self.points, dtype=np.float32)

    def _on_press(self, event) -> None:
        if self.ax is None or event.button not in {1, MouseButton.LEFT}:
            return
        point = self._event_to_data_point(event)
        if point is None:
            return
        point_index = self._find_point_index(event)
        if point_index is not None:
            self.active_index = point_index
            return
        if len(self.points) >= self.count:
            return
        self.points.append(point)
        self.active_index = len(self.points) - 1
        self._refresh()

    def _on_motion(self, event) -> None:
        if self.ax is None or self.active_index is None:
            return
        point = self._event_to_data_point(event)
        if point is None:
            return
        self.points[self.active_index] = point
        self._refresh()

    def _on_release(self, _event) -> None:
        self.active_index = None

    def _on_confirm(self, _event) -> None:
        if len(self.points) != self.count:
            self._set_status(f"Add {self.count - len(self.points)} more point(s) before confirming.")
            return
        self.confirmed = True
        plt.close(self.fig)

    def _find_point_index(self, event) -> int | None:
        if not self.points or self.ax is None:
            return None
        pixel_points = self.ax.transData.transform(np.vstack(self.points))
        distances = np.linalg.norm(pixel_points - np.array([event.x, event.y]), axis=1)
        closest = int(np.argmin(distances))
        if float(distances[closest]) <= _POINT_PICK_RADIUS_PX:
            return closest
        return None

    def _event_to_data_point(self, event) -> np.ndarray | None:
        if self.ax is None or event.x is None or event.y is None:
            return None
        if not self.ax.bbox.contains(event.x, event.y):
            return None
        x_data, y_data = self.ax.transData.inverted().transform((event.x, event.y))
        return np.array([x_data, y_data], dtype=np.float32)

    def _refresh(self) -> None:
        if self.current_scatter is None or self.fig is None:
            return
        offsets = np.asarray(self.points, dtype=np.float32) if self.points else np.empty((0, 2), dtype=np.float32)
        self.current_scatter.set_offsets(offsets)
        for annotation in self.point_annotations:
            annotation.remove()
        self.point_annotations = []
        for index, point in enumerate(self.points):
            self.point_annotations.append(
                self.ax.annotate(
                    self.point_labels[index],
                    xy=(float(point[0]), float(point[1])),
                    xytext=(8, -12),
                    textcoords="offset points",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "#ffd166", "ec": "black", "alpha": 0.9},
                )
            )
        if len(self.points) == self.count:
            self._set_status("Drag points to adjust them, then click Confirm.")
        else:
            self._set_status(f"Click to place {self.count - len(self.points)} more point(s). You can drag placed points before confirming.")
        self.fig.canvas.draw_idle()

    def _set_status(self, message: str) -> None:
        if self.status_text is not None:
            self.status_text.set_text(_wrap_display_text(message, _STATUS_WRAP_WIDTH))


def _capture_distance(
    image: CalibratedImage,
    label: str,
    instructions: str,
    point_labels: tuple[str, str] = ("Point 1", "Point 2"),
    existing_landmarks: list[Landmark] | None = None,
) -> tuple[float, list[Landmark]]:
    points = _click_points(
        image.image,
        2,
        f"{label}: {instructions}",
        existing_landmarks=existing_landmarks,
        point_labels=point_labels,
    )
    rectified_points = _transform_points(points, image.transform)
    return _pixel_distance_cm(rectified_points[0], rectified_points[1]), [
        Landmark(point_labels[0], points[0]),
        Landmark(point_labels[1], points[1]),
    ]


def _capture_center(
    image: CalibratedImage,
    title: str,
    instructions: str,
    existing_landmarks: list[Landmark] | None = None,
) -> tuple[np.ndarray, Landmark]:
    point = _click_points(
        image.image,
        1,
        f"{title}: {instructions}",
        existing_landmarks=existing_landmarks,
        point_labels=("Head center",),
    )
    return _transform_points(point, image.transform)[0], Landmark("Head center", point[0])


def _capture_center_distance(
    image: CalibratedImage,
    center: np.ndarray,
    label: str,
    instructions: str,
    point_label: str = "Target point",
    existing_landmarks: list[Landmark] | None = None,
) -> tuple[float, Landmark]:
    point = _click_points(
        image.image,
        1,
        f"{label}: {instructions} Reference point: the head center you already clicked in the side image.",
        existing_landmarks=existing_landmarks,
        point_labels=(point_label,),
    )
    rectified_point = _transform_points(point, image.transform)[0]
    return _pixel_distance_cm(center, rectified_point), Landmark(point_label, point[0])


def _capture_numeric(label: str, instructions: str) -> float:
    prompt = f"{label} - {instructions}\nEnter numeric value in inches: "
    return float(input(prompt).strip())


def _pixel_distance_cm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) * 2.54 / 100.0)
