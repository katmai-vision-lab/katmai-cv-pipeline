"""
src/environment/video_context.py

Utility for resolving the datetime and location of a video — needed before
making NPS/NOAA environmental data API calls.

- Camera recordings (phone/DSLR): datetime and GPS extracted from metadata
  automatically; user is prompted only for any missing fields.
- Screen recordings / downloads: user supplies datetime and location manually.

Dependencies: pymediainfo (bundles libmediainfo in wheel), mutagen (pure Python)
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BROOKS_FALLS_LAT = 58.4788
BROOKS_FALLS_LON = -155.0636

# Heuristic score: positive → camera, negative → screen recording
_SIGNALS = {
    "has_gps":                  +40,
    "has_make_model":           +25,
    "has_quicktime_tags":       +20,
    "hevc_codec":               +10,
    "h264_codec":               +5,
    "lavf_encoder":             -40,
    "ffmpeg_encoder":           -35,
    "yt_dlp_comment":           -50,
    "youtube_url_comment":      -40,
    "vp9_codec":                -30,
    "av1_codec":                -30,
    "obs_software":             -50,
    "screen_recorder_software": -50,
}


@dataclass
class VideoContext:
    video_path: str
    source_type: str            # "camera" | "screen_recording" | "unknown"
    datetime_utc: Optional[str] # ISO 8601 UTC string
    latitude: Optional[float]
    longitude: Optional[float]
    detection_confidence: str   # "high" | "medium" | "low"
    detection_score: int
    detection_notes: list


# ── Metadata readers ──────────────────────────────────────────────────────────

def _mediainfo(path: str) -> dict:
    from pymediainfo import MediaInfo
    info = MediaInfo.parse(path)
    out = {}
    for track in info.tracks:
        d = track.to_data()
        if track.track_type == "Video":
            out["codec"] = (d.get("format") or "").lower()
        if track.track_type == "General":
            out.update({k: d.get(k) or "" for k in (
                "writing_application", "writing_library", "comment",
                "encoded_date", "tagged_date", "gps_position",
            )})
            out["make"]  = d.get("comapplefiles_make") or d.get("make") or ""
            out["model"] = d.get("comapplefiles_model") or d.get("model") or ""
    return out


def _mutagen_tags(path: str) -> dict:
    try:
        from mutagen.mp4 import MP4
        f = MP4(path)
        tags = f.tags or {}
    except Exception:
        return {}

    def first(key):
        v = tags.get(key)
        if v and not isinstance(v, str):
            v = v[0]
        return str(v).strip() if v else ""

    qt_keys = [k for k in tags if "com.apple.quicktime" in str(k).lower()]
    return {
        "encoder": first("©too") or first("©swr"),
        "datetime_str": first("©day"),
        "gps_str": first("©xyz"),
        "comment": first("©cmt"),
        "make": first("©mak"),
        "model": first("©mod"),
        "qt_key_count": len(qt_keys),
    }


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_iso6709(s: str) -> Optional[tuple[float, float]]:
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", s.strip())
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None


def _parse_dt(s: str) -> Optional[datetime]:
    s = re.sub(r"^UTC\s+", "", s.strip())
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── Classification ────────────────────────────────────────────────────────────

def _classify(mi: dict, mt: dict) -> tuple[str, int, list[str]]:
    score, notes = 0, []

    codec   = mi.get("codec", "")
    encoder = (mi.get("writing_application", "") or mt.get("encoder", "")).lower()
    comment = (mi.get("comment", "") or mt.get("comment", "")).lower()
    gps_str = mi.get("gps_position", "") or mt.get("gps_str", "")

    def apply(key, condition, note):
        nonlocal score
        if condition:
            score += _SIGNALS[key]
            notes.append(note)

    apply("has_gps",          gps_str and _parse_iso6709(gps_str),
          "GPS coordinates present in metadata")
    apply("has_make_model",   mi.get("make") or mi.get("model") or mt.get("make") or mt.get("model"),
          f"Camera make/model: {(mi.get('make') or mt.get('make', ''))} {(mi.get('model') or mt.get('model', ''))}".strip())
    apply("has_quicktime_tags", mt.get("qt_key_count", 0) > 0,
          f"Apple QuickTime atoms present ({mt['qt_key_count']})")
    apply("hevc_codec",       codec in ("hevc", "h265"),   f"Codec: {codec} (phone camera)")
    apply("h264_codec",       codec in ("avc", "h264"),    f"Codec: {codec}")
    apply("vp9_codec",        codec == "vp9",              "Codec: VP9 (YouTube streaming)")
    apply("av1_codec",        codec == "av1",              "Codec: AV1 (YouTube/web streaming)")
    apply("lavf_encoder",     "lavf" in encoder or "libav" in encoder,
          f"Encoder: '{encoder}' — ffmpeg/libav (screen recording or re-encode)")
    apply("ffmpeg_encoder",   "ffmpeg" in encoder and "lavf" not in encoder,
          f"Encoder: '{encoder}' — ffmpeg")
    apply("obs_software",     "obs" in encoder or "obs" in mi.get("writing_library", "").lower(),
          "OBS Studio detected")
    apply("screen_recorder_software",
          any(s in encoder for s in ("screenflow", "camtasia", "bandicam", "screen capture")),
          f"Screen recorder software: '{encoder}'")
    apply("yt_dlp_comment",   any(s in comment for s in ("yt-dlp", "ytdl", "youtube-dl")),
          "Comment tag contains yt-dlp / youtube-dl marker")
    apply("youtube_url_comment", "youtube.com" in comment or "youtu.be" in comment,
          "Comment tag contains YouTube URL")

    source = "camera" if score >= 15 else "screen_recording" if score <= -15 else "unknown"
    return source, score, notes


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_video(
    video_path: str,
    non_interactive: bool = False,
    supplied_datetime: Optional[str] = None,
    supplied_location: Optional[str] = None,
) -> VideoContext:
    """
    Classify a video and resolve datetime + location for environmental API calls.

    Args:
        video_path:        Path to the video file.
        non_interactive:   Skip prompts; exit if required fields are missing.
        supplied_datetime: Fallback datetime string, e.g. "2024-07-15 14:30" (UTC).
        supplied_location: Fallback location string, e.g. "58.4788,-155.0636".
    """
    path = Path(video_path)
    if not path.exists():
        print(f"[error] File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    mi = _mediainfo(str(path))
    mt = _mutagen_tags(str(path))
    source_type, score, notes = _classify(mi, mt)
    confidence = "high" if abs(score) >= 30 else "medium" if abs(score) >= 15 else "low"

    print(f"\nAnalyzing: {path.name}")
    print(f"  Source : {source_type.upper().replace('_', ' ')}  "
          f"(score {score:+d}, confidence: {confidence})")
    for note in notes:
        print(f"    • {note}")

    dt: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    if source_type == "camera":
        # Try to auto-extract from metadata
        raw_dt = mt.get("datetime_str") or mi.get("encoded_date") or mi.get("tagged_date") or ""
        if raw_dt:
            dt = _parse_dt(raw_dt)
            if dt and dt.year <= 1970:
                dt = None  # skip placeholder epochs

        gps = _parse_iso6709(mi.get("gps_position", "") or mt.get("gps_str", "") or "")
        if gps:
            lat, lon = gps

    # Resolve any missing fields via supplied args or interactive prompts
    def _resolve_dt():
        nonlocal dt
        if supplied_datetime:
            dt = _parse_dt(supplied_datetime)
            if not dt:
                print(f"[error] Cannot parse datetime: {supplied_datetime!r}", file=sys.stderr)
                sys.exit(1)
        elif not non_interactive:
            print("\n  Enter recording datetime (UTC). Format: YYYY-MM-DD HH:MM")
            while not dt:
                dt = _parse_dt(input("  Datetime: ").strip())
                if not dt:
                    print("  [!] Try: 2024-07-15 14:30")
        else:
            print("[error] --datetime required in non-interactive mode.", file=sys.stderr)
            sys.exit(1)

    def _resolve_loc():
        nonlocal lat, lon
        if supplied_location:
            parts = re.split(r"[,;]", supplied_location)
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
        elif not non_interactive:
            print(f"\n  Enter location (lat,lon). Press Enter for Brooks Falls default "
                  f"({BROOKS_FALLS_LAT}, {BROOKS_FALLS_LON}).")
            raw = input("  Lat, Lon: ").strip()
            if not raw:
                lat, lon = BROOKS_FALLS_LAT, BROOKS_FALLS_LON
            else:
                parts = re.split(r"[,;]", raw)
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
        else:
            print("[error] --location required in non-interactive mode.", file=sys.stderr)
            sys.exit(1)

    if dt is None:
        _resolve_dt()
    if lat is None:
        _resolve_loc()

    return VideoContext(
        video_path=str(path.resolve()),
        source_type=source_type,
        datetime_utc=dt.isoformat() if dt else None,
        latitude=lat,
        longitude=lon,
        detection_confidence=confidence,
        detection_score=score,
        detection_notes=notes,
    )
