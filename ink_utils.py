"""
ink_utils.py

1:1-Portierung der Wort-Segmentierung und Ink-Normalisierung aus
OcrStrokeCollector.ts (Plugin-Quellcode). Bewusst so nah wie möglich am
TypeScript-Original gehalten, damit Training und Laufzeit-Inferenz exakt
dieselbe Segmentierung/Normalisierung anwenden. Bei Änderungen IMMER
gegenprüfen, ob die TS-Version im Plugin mitgezogen werden muss (und
umgekehrt).

Ein "Stroke" ist hier eine Liste von Punkten: [{'x': float, 'y': float, 't': float}, ...]
"""
import math
import random
from typing import List, Dict, Tuple

Point = Dict[str, float]
Stroke = List[Point]
Bounds = Dict[str, float]


# Zeichen, die im Training als eigenständige Zielklassen zugelassen sind.
# Alles andere wird beim Aufbau der Trainingsdaten aus dem ZIELTEXT entfernt
# (siehe normalize_text_for_training) -- die zugehörigen Federstriche
# bleiben aber im Eingabesignal (Ink) erhalten. Das CTC-Training lernt so
# explizit, für diese Formen KEINE Zeichenausgabe (Blank) zu erzeugen,
# statt sie fälschlich einem Buchstaben zuzuordnen.
ALLOWED_BASE_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Umlaute/ß -> Basisform. Wird VOR dem Filtern auf ALLOWED_BASE_CHARS
# angewendet, sodass z. B. "ä" durchgängig als "a" behandelt wird: dieselbe
# Zielklasse trotz optisch anderer Handschrift-Form (a + zwei Punkte). Das
# Modell muss den Unterschied dadurch nicht als eigene Klasse lernen.
UMLAUT_MAP = {
    "ä": "a", "ö": "o", "ü": "u",
    "Ä": "A", "Ö": "O", "Ü": "U",
    "ß": "ss",
}


def normalize_text_for_training(text: str) -> str:
    """
    Wandelt einen rohen Wort-Text in die tatsächliche CTC-Zielsequenz um:
      1. Umlaute/ß über UMLAUT_MAP auf ihre Basisform abbilden.
      2. Alle Zeichen entfernen, die nicht in ALLOWED_BASE_CHARS liegen
         (Satzzeichen, Leerzeichen, sonstige Sonderzeichen).
    Das Ergebnis kann leer sein (z. B. bei einem isolierten Satzzeichen) --
    das ist ein gültiges, sogar nützliches Trainingsbeispiel: es lehrt das
    Modell, für rein "unbekannt geformte" Striche durchgängig Blank
    auszugeben, statt zu raten.
    """
    mapped = "".join(UMLAUT_MAP.get(c, c) for c in text)
    return "".join(c for c in mapped if c in ALLOWED_BASE_CHARS)

def stroke_bounds(stroke: Stroke) -> Bounds:
    xs = [p["x"] for p in stroke]
    ys = [p["y"] for p in stroke]
    if not xs:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {"x": min_x, "y": min_y, "width": max(0.0, max_x - min_x), "height": max(0.0, max_y - min_y)}


def merge_bounds(bounds_list: List[Bounds]) -> Bounds:
    min_x = min(b["x"] for b in bounds_list)
    min_y = min(b["y"] for b in bounds_list)
    max_x = max(b["x"] + b["width"] for b in bounds_list)
    max_y = max(b["y"] + b["height"] for b in bounds_list)
    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def average_stroke_height(strokes: List[Stroke]) -> float:
    if not strokes:
        return 10.0
    heights = [stroke_bounds(s)["height"] or 1.0 for s in strokes]
    return sum(heights) / len(heights)


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _interval_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 != 0:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _interval_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0

FRAGMENT_MAX_STROKES = 2  # Cluster mit höchstens dieser Strichanzahl gelten als "Fragment" (siehe TS-Original)


def _cluster_bounds_distance(a: Bounds, b: Bounds, line_gap: float, word_gap: float) -> float:
    """Entspricht clusterBoundsDistance() in OcrStrokeCollector.ts."""
    a_center_y = a["y"] + a["height"] / 2
    b_center_y = b["y"] + b["height"] / 2
    v_dist = abs(a_center_y - b_center_y)
    h_gap = _interval_gap(a["x"], a["x"] + a["width"], b["x"], b["x"] + b["width"])
    return max(v_dist / line_gap, h_gap / word_gap)

def group_strokes_into_words(
    strokes: List[Stroke],
    line_gap_factor: float = 1.6,
    word_gap_factor: float = 1.2,
    fragment_rescue_factor: float = 1.6,
) -> List[Dict]:
    """
    Entspricht groupStrokesIntoWords() in OcrStrokeCollector.ts, inklusive
    des zweiten Rettungs-Durchlaufs für kleine, isoliert gebliebene
    Fragmente (siehe dortige Kommentare für die Begründung).
    """
    if not strokes:
        return []

    bounds_list = [stroke_bounds(s) for s in strokes]
    heights = [b["height"] or 1.0 for b in bounds_list]
    widths = [b["width"] or 1.0 for b in bounds_list]
    median_height = _median(heights) or 1.0
    median_width = _median(widths) or 1.0

    line_gap = median_height * line_gap_factor
    word_gap = median_width * word_gap_factor

    def cy(i: int) -> float:
        b = bounds_list[i]
        return b["y"] + b["height"] / 2

    order = sorted(range(len(strokes)), key=cy)
    uf = _DisjointSet(len(strokes))
    vertical_window = line_gap + median_height

    # ---- Durchlauf 1: strenge, paarweise Clusterung ----
    for a in range(len(order)):
        ia = order[a]
        ba = bounds_list[ia]
        for b in range(a + 1, len(order)):
            ib = order[b]
            if cy(ib) - cy(ia) > vertical_window:
                break

            v_center_dist = abs(cy(ib) - cy(ia))
            if v_center_dist > line_gap:
                continue

            bb = bounds_list[ib]
            h_gap = _interval_gap(ba["x"], ba["x"] + ba["width"], bb["x"], bb["x"] + bb["width"])
            if h_gap > word_gap:
                continue

            uf.union(ia, ib)

    # ---- Durchlauf 2: kleine, isolierte Fragmente der naheliegendsten
    # Bounding Box zuordnen ----
    if fragment_rescue_factor > 1.0:
        max_iterations = len(strokes)
        for _ in range(max_iterations):
            cluster_map: Dict[int, List[int]] = {}
            for i in range(len(strokes)):
                cluster_map.setdefault(uf.find(i), []).append(i)

            entries = [
                {"root": root, "indices": indices, "bounds": merge_bounds([bounds_list[i] for i in indices])}
                for root, indices in cluster_map.items()
            ]

            merged = False
            for candidate in entries:
                if len(candidate["indices"]) > FRAGMENT_MAX_STROKES:
                    continue

                best_other = None
                best_dist = float("inf")
                for other in entries:
                    if other["root"] == candidate["root"]:
                        continue
                    dist = _cluster_bounds_distance(candidate["bounds"], other["bounds"], line_gap, word_gap)
                    if dist < best_dist:
                        best_dist = dist
                        best_other = other

                if best_other is not None and best_dist <= fragment_rescue_factor:
                    uf.union(candidate["indices"][0], best_other["indices"][0])
                    merged = True
                    break

            if not merged:
                break

    clusters: Dict[int, List[int]] = {}
    for i in range(len(strokes)):
        clusters.setdefault(uf.find(i), []).append(i)

    return [
        {
            "strokes": [strokes[i] for i in indices],
            "bounds": merge_bounds([bounds_list[i] for i in indices]),
        }
        for indices in clusters.values()
    ]


def ramer_douglas_peucker(points: Stroke, tolerance: float) -> Stroke:
    """
    1:1-Portierung von douglasPeucker()/simplifyStroke() in StrokeSimplify.ts.
    Arbeitet auf rohen {'x','y',...}-Punkt-Dicts; behaelt bei den erhaltenen
    Punkten alle Original-Felder (z. B. 't') unveraendert bei, es wird nichts
    interpoliert -- reine Teilmengenauswahl, exakt wie im TS-Original.
    """
    if len(points) < 3:
        return points

    def perpendicular_distance(pt: Point, line_start: Point, line_end: Point) -> float:
        dx = line_end["x"] - line_start["x"]
        dy = line_end["y"] - line_start["y"]
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.hypot(pt["x"] - line_start["x"], pt["y"] - line_start["y"])
        t = ((pt["x"] - line_start["x"]) * dx + (pt["y"] - line_start["y"]) * dy) / length_sq
        proj_x = line_start["x"] + t * dx
        proj_y = line_start["y"] + t * dy
        return math.hypot(pt["x"] - proj_x, pt["y"] - proj_y)

    max_dist, max_index = 0.0, 0
    first, last = points[0], points[-1]
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], first, last)
        if dist > max_dist:
            max_dist, max_index = dist, i

    if max_dist > tolerance:
        left = ramer_douglas_peucker(points[: max_index + 1], tolerance)
        right = ramer_douglas_peucker(points[max_index:], tolerance)
        return left[:-1] + right
    return [first, last]


def simplify_stroke_random_tolerance(
    stroke: Stroke,
    bounds: Bounds,
    rng: random.Random,
    max_tolerance_fraction: float = 0.05,
) -> Stroke:
    """
    Wendet ramer_douglas_peucker() mit einer zufaellig gezogenen, relativ zur
    Wort-Diagonale skalierten Toleranz an (Data Augmentation). Simuliert die
    Bandbreite moeglicher settings.strokeSimplifyTolerance-Werte im Plugin
    (0-3 PDF-Punkte), OHNE von der absoluten Groesse der Trainingsbeispiele
    abhaengig zu sein -- entscheidend, da die im Plugin gespeicherten
    Striche je nach Handschriftgroesse sehr unterschiedliche absolute
    Ausmasse haben koennen.
    """
    diagonal = math.hypot(bounds["width"], bounds["height"]) or 1.0
    tolerance_fraction = rng.uniform(0.0, max_tolerance_fraction)
    tolerance = tolerance_fraction * diagonal
    if tolerance <= 0:
        return stroke
    return ramer_douglas_peucker(stroke, tolerance)


def resample_normalized_points(points: List[Point], spacing: float) -> List[Point]:
    """
    1:1-Portierung von resampleNormalizedPoints() in OcrStrokeCollector.ts.
    points: Liste von {'x','y','t'}, bereits auf die Wort-Bounding-Box
    normiert (x,y typischerweise in [0,1]). Interpoliert linear zusaetzliche
    Zwischenpunkte, sodass aufeinanderfolgende Punkte hoechstens `spacing`
    (in denselben normierten Einheiten) auseinanderliegen.
    """
    if len(points) < 2 or spacing <= 0:
        return points

    result: List[Point] = [points[0]]
    carry = 0.0
    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        seg_length = math.hypot(dx, dy)
        if seg_length == 0:
            continue
        distance_along = spacing - carry
        while distance_along < seg_length:
            frac = distance_along / seg_length
            result.append({
                "x": prev["x"] + dx * frac,
                "y": prev["y"] + dy * frac,
                "t": prev["t"] + (curr["t"] - prev["t"]) * frac,
            })
            distance_along += spacing
        carry = distance_along - seg_length

    last = points[-1]
    if result[-1] != last:
        result.append(last)
    return result


def build_normalized_ink(
    strokes: List[Stroke],
    bounds: Bounds,
    max_seq_len: int,
    resample_spacing: float = 0.0,
) -> Tuple[List[List[float]], int]:
    """
    Entspricht buildNormalizedInk() in OcrStrokeCollector.ts.
    Gibt (gepaddete_sequenz, tatsaechliche_laenge_vor_padding) zurueck.

    resample_spacing > 0: interpoliert NACH der Normalisierung zusaetzliche
    Punkte pro Strich (siehe resample_normalized_points), damit die
    Punktdichte vereinfachter (Douglas-Peucker-reduzierter) Striche wieder
    an die zur Laufzeit im Plugin erzeugte Dichte angeglichen wird.
    0 = kein Resampling (Original-Verhalten vor dieser Erweiterung).
    """
    all_t = [p["t"] for stroke in strokes for p in stroke]
    min_t, max_t = min(all_t), max(all_t)
    t_span = max(1.0, max_t - min_t)
    scale = max(bounds["width"], bounds["height"], 1.0)

    all_points: List[Point] = []
    for stroke in strokes:
        normalized = [
            {
                "x": (p["x"] - bounds["x"]) / scale,
                "y": (p["y"] - bounds["y"]) / scale,
                "t": (p["t"] - min_t) / t_span,
            }
            for p in stroke
        ]
        if resample_spacing > 0:
            normalized = resample_normalized_points(normalized, resample_spacing)
        all_points.extend(normalized)
    all_points.sort(key=lambda p: p["t"])

    valid_length = min(len(all_points), max_seq_len)
    seq = [[p["x"], p["y"], p["t"]] for p in all_points[:max_seq_len]]
    while len(seq) < max_seq_len:
        seq.append([0.0, 0.0, 0.0])
    return seq, valid_length