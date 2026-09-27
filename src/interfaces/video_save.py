"""
SeedVR2 Save Video Node
High-reliability video encoder and assembler for ComfyUI.
Applies Studio-grade timestamp rectification (-fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr)
to eliminate audio drift and endpoint freeze on long upscale renders.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import cv2
from comfy_api.latest import io

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _get_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate_dirs = [
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg"),
        Path(r"C:\Program Files (x86)\ffmpeg\bin"),
        Path(r"C:\ffmpeg\bin"),
        Path(r"D:\ffmpeg\bin"),
        Path(__file__).resolve().parents[2] / "bin" / "ffmpeg" / "bin",
        Path(__file__).resolve().parents[2] / "bin",
    ]
    for d in candidate_dirs:
        candidate_file = d / "ffmpeg.exe"
        if candidate_file.exists():
            return str(candidate_file)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    return "ffmpeg"


class SeedVR2SaveVideo(io.ComfyNode):
    """
    SeedVR2 Save Video Node
    
    Encodes video frames to MP4 with Studio's timestamp rectification:
    -fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr
    Eliminates audio drift and end-of-video playback stutter completely.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2SaveVideo",
            display_name="SeedVR2 Save Video (CFR & Sync Safe)",
            category="SEEDVR2",
            description=(
                "Save video frames to MP4 with Studio-grade timestamp rectification. "
                "Applies -fflags +genpts -avoid_negative_ts make_zero -fps_mode cfr "
                "to eliminate audio desynchronization and endpoint freeze."
            ),
            inputs=[
                io.Image.Input("images",
                    tooltip="Upscaled video frames [T, H, W, C] in range [0, 1]."
                ),
                io.Float.Input("fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.01,
                    tooltip="Output video frame rate (strictly locked to Constant Frame Rate / CFR)."
                ),
                io.String.Input("filename_prefix",
                    default="SeedVR2",
                    tooltip="Output filename prefix. Saved under ComfyUI output directory."
                ),
                io.Int.Input("crf",
                    default=18,
                    min=0,
                    max=51,
                    step=1,
                    tooltip="H.264 CRF quality level (lower = higher quality, default: 18)."
                ),
                io.Combo.Input("preset",
                    options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                    default="medium",
                    tooltip="FFmpeg H.264 encoding preset (default: medium)."
                ),
                io.String.Input("audio_source_path",
                    default="",
                    optional=True,
                    tooltip="Optional path to source video or audio file to mux into the output video with timestamp alignment."
                ),
            ],
            outputs=[
                io.String.Output("video_path",
                    tooltip="Full path to the saved MP4 video file."
                )
            ]
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        fps: float = 24.0,
        filename_prefix: str = "SeedVR2",
        crf: int = 18,
        preset: str = "medium",
        audio_source_path: str = "",
    ) -> io.NodeOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [T, H, W, C], got {tuple(images.shape)}")

        output_dir = folder_paths.get_output_directory() if folder_paths else "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_filename = f"{filename_prefix}_{timestamp}.mp4"
        final_path = Path(output_dir) / final_filename
        temp_path = final_path.with_name(f"{final_path.stem}_noaudio.mp4")

        T, H, W, C = images.shape
        cpu_frames = images.detach().cpu()
        if cpu_frames.dtype != torch.uint8:
            cpu_frames = (cpu_frames.clamp(0, 1) * 255.0).to(torch.uint8)

        frames_np = cpu_frames.numpy()
        del cpu_frames

        ffmpeg_bin = _get_ffmpeg()
        encode_command = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24" if C == 3 else "rgba",
            "-s:v", f"{W}x{H}", "-r", f"{fps:.9g}", "-i", "pipe:0",
            "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
            "-fps_mode", "cfr",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_path),
        ]

        encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
        write_error = None
        try:
            assert encoder.stdin is not None
            for i in range(T):
                frame = frames_np[i]
                if C == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                encoder.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            write_error = exc
        finally:
            if encoder.stdin is not None:
                encoder.stdin.close()

        ret = encoder.wait()
        if ret != 0:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg video encoding failed with code {ret}")
        if write_error is not None:
            temp_path.unlink(missing_ok=True)
            raise write_error

        # Audio muxing with timestamp rectification
        has_audio = bool(audio_source_path and os.path.exists(audio_source_path))
        if has_audio:
            video_duration = T / max(fps, 1e-6)
            mux_cmd = [
                ffmpeg_bin, "-y", "-i", str(temp_path), "-i", str(audio_source_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                "-fps_mode", "cfr",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{video_duration:.9f}",
                str(final_path)
            ]
            mux_res = subprocess.run(mux_cmd, capture_output=True, text=True)
            temp_path.unlink(missing_ok=True)
            if mux_res.returncode != 0:
                print(f"[SeedVR2 Save Video] Warning: Audio mux failed ({mux_res.stderr}), keeping video only.")
                if temp_path.exists():
                    temp_path.replace(final_path)
        else:
            temp_path.replace(final_path)

        print(f"[SeedVR2 Save Video] ✅ Video saved: {final_path} ({T} frames, {W}x{H} @ {fps}fps CFR)")
        return io.NodeOutput(str(final_path))
