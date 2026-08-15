"""Multi-page PDF inspection report builder using matplotlib's PdfPages.

Pages produced (preserving the source-notebook layout):
    1. Title page — station, camera, direction, rake type, video link.
    2. Wagon-status summary — top table (totals) + bottom table (per-wagon rows).
    3. Problem-frame section — only if confirmed damage / door issues exist.
    4. Locomotive grid — auto-sized grid of loco middle frames.
    5. Individual wagon pages — 3 representative frames per wagon (25%/55%/80%)
       or problem frames where applicable.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class CameraStyle:
    """Per-camera display strings + direction→rake-type mapping."""

    camera_label: str = "CAMERA"
    station_name: str = "HAZARIBAGH"
    flavour: str = "top"  # "top" | "side"
    # Direction → (rake_type, color, arrow_symbol)
    # Defaults match the side-camera convention; top-right inverts via config.
    direction_to_rake: dict[str, tuple[str, str, str]] = field(default_factory=lambda: {
        "left-to-right": ("LOADED RAKE", "darkblue", "→"),
        "right-to-left": ("EMPTY RAKE", "darkgreen", "←"),
    })

    def rake_for(self, direction: str) -> tuple[str, str, str]:
        return self.direction_to_rake.get(direction, ("DIRECTION UNKNOWN", "gray", "↔"))


class PdfReportBuilder:
    """Produces a multi-page inspection PDF for one trimmed-train clip."""

    PAGE_FIGSIZE = (11.69, 8.27)  # A4 landscape
    WAGON_ROWS_PER_PAGE = 20
    PROBLEM_FRAMES_PER_PAGE = 3
    DEFAULT_POSITIONS = (0.25, 0.55, 0.80)

    def __init__(
        self,
        style: CameraStyle,
        logo_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        positions: Optional[tuple[float, ...]] = None,
    ):
        self.style = style
        self.logo_path = logo_path
        self.logger = logger or logging.getLogger(__name__)
        self.positions = tuple(positions or self.DEFAULT_POSITIONS)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        output_path: str,
        raw_video_name: str,
        upload_timestamp: datetime,
        direction: str,
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
        loco_summary_df: pd.DataFrame,
        problem_frames_df: pd.DataFrame,
        trimmed_video_url: Optional[str] = None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with PdfPages(output_path) as pdf:
            self._title_page(
                pdf, raw_video_name=raw_video_name, upload_timestamp=upload_timestamp,
                direction=direction, trimmed_video_url=trimmed_video_url,
                segment_summary_df=segment_summary_df,
                damage_results_df=damage_results_df,
            )
            self._summary_pages(pdf, segment_summary_df, damage_results_df,
                                upload_timestamp=upload_timestamp)
            self._problem_pages(pdf, segment_summary_df, damage_results_df,
                                problem_frames_df)
            self._loco_pages(pdf, loco_summary_df)
            self._wagon_pages(pdf, segment_summary_df, damage_results_df,
                              problem_frames_df)

        pdf_metadata = self._pdf_metadata(raw_video_name, upload_timestamp)
        self.logger.info("PDF written: %s (%d bytes)", output_path,
                         os.path.getsize(output_path))
        return output_path

    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_metadata(raw_video_name: str, upload_timestamp: datetime) -> dict:
        return {
            "Title": f"Train Inspection Report — {raw_video_name}",
            "Author": "Automated CCTV Analytics System",
            "Subject": f"Inspection report generated at {upload_timestamp.isoformat()}",
        }

    # ------------------------------------------------------------------
    # title page
    # ------------------------------------------------------------------

    def _title_page(
        self,
        pdf: PdfPages,
        *,
        raw_video_name: str,
        upload_timestamp: datetime,
        direction: str,
        trimmed_video_url: Optional[str],
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
    ) -> None:
        rake_type, rake_color, arrow = self.style.rake_for(direction)
        fig = plt.figure(figsize=self.PAGE_FIGSIZE)
        fig.suptitle(
            "TRAIN INSPECTION REPORT",
            fontsize=22, fontweight="bold", y=0.95,
        )
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
        ax.axis("off")

        if self.logo_path and os.path.exists(self.logo_path):
            try:
                logo = plt.imread(self.logo_path)
                fig.figimage(logo, xo=40, yo=fig.bbox.height - 80, alpha=0.9)
            except Exception:  # noqa: BLE001
                pass

        lines = [
            ("Station", self.style.station_name),
            ("Camera", self.style.camera_label),
            ("Rake Type", f"{rake_type}  {arrow}"),
            ("Direction", direction),
            ("Raw Video", raw_video_name),
            ("Generated", upload_timestamp.strftime("%d-%m-%Y %H:%M:%S")),
        ]
        text_y = 0.85
        for label, value in lines:
            ax.text(0.05, text_y, f"{label}:", fontsize=14, fontweight="bold")
            color = rake_color if label == "Rake Type" else "black"
            ax.text(0.30, text_y, str(value), fontsize=14, color=color)
            text_y -= 0.07

        if trimmed_video_url:
            ax.text(
                0.05, text_y - 0.05,
                "Trimmed Video:", fontsize=14, fontweight="bold",
            )
            ax.text(
                0.30, text_y - 0.05, trimmed_video_url,
                fontsize=10, color="blue", url=trimmed_video_url, wrap=True,
            )

        total_wagons = int(
            (segment_summary_df["segment_type"].isin({"wagon", "wagon_loaded"})).sum()
            if "segment_type" in segment_summary_df.columns else len(segment_summary_df)
        )
        damaged = int(damage_results_df["damage_detected"].sum()) if not damage_results_df.empty else 0
        ax.text(
            0.05, 0.2,
            f"Total Wagons: {total_wagons}    Damaged: {damaged}",
            fontsize=16, fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # summary page(s)
    # ------------------------------------------------------------------

    def _wagon_status_row(self, seg_id: int, damage_row: dict) -> tuple[str, str]:
        if self.style.flavour == "top":
            if damage_row.get("damage_detected"):
                return "DAMAGE", "#ff5252"
            if damage_row.get("probable_damage_detected"):
                return "PROBABLE DAMAGE", "#ffb74d"
            return "OK", "#81c784"
        # side flavour
        if damage_row.get("damage_detected"):
            return "DAMAGE", "#ff5252"
        if damage_row.get("door_status") == "open":
            return "DOOR OPEN", "#ffb74d"
        return "OK", "#81c784"

    def _summary_pages(
        self,
        pdf: PdfPages,
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
        upload_timestamp: datetime,
    ) -> None:
        if segment_summary_df.empty:
            return

        damage_lookup = (
            damage_results_df.set_index("wagon_id").to_dict("index")
            if not damage_results_df.empty else {}
        )

        wagons = segment_summary_df[
            segment_summary_df["segment_type"].isin({"wagon", "wagon_loaded"})
        ].reset_index(drop=True) if "segment_type" in segment_summary_df.columns \
            else segment_summary_df.copy()

        total = len(wagons)
        damaged = sum(
            1 for sid in wagons["segment_id"]
            if damage_lookup.get(int(sid), {}).get("damage_detected")
        )

        rows_per_page = self.WAGON_ROWS_PER_PAGE
        pages = max(1, math.ceil(len(wagons) / rows_per_page))

        for page_idx in range(pages):
            fig = plt.figure(figsize=self.PAGE_FIGSIZE)
            fig.suptitle(
                f"WAGON STATUS SUMMARY (Page {page_idx + 1} of {pages})",
                fontsize=16, fontweight="bold",
            )
            ax_top = fig.add_subplot(2, 1, 1)
            ax_top.axis("off")
            ax_top.table(
                cellText=[[
                    upload_timestamp.strftime("%d-%m-%Y %H:%M"),
                    total, damaged,
                    "ISSUES" if damaged else "OK",
                ]],
                colLabels=["DATE-TIME", "TOTAL WAGONS", "DAMAGED WAGONS", "STATUS"],
                loc="center", cellLoc="center",
            )

            slice_df = wagons.iloc[page_idx * rows_per_page:(page_idx + 1) * rows_per_page]
            cell_text = []
            cell_colors = []
            for sr_no, (_, row) in enumerate(slice_df.iterrows(), start=page_idx * rows_per_page + 1):
                seg_id = int(row["segment_id"])
                status, color = self._wagon_status_row(seg_id, damage_lookup.get(seg_id, {}))
                cell_text.append([sr_no, seg_id, status])
                cell_colors.append(["white", "white", color])
            ax_bot = fig.add_subplot(2, 1, 2)
            ax_bot.axis("off")
            tbl = ax_bot.table(
                cellText=cell_text or [["—", "—", "—"]],
                colLabels=["SR.NO", "WAGON ID", "STATUS"],
                cellColours=cell_colors or None,
                loc="center", cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # problem-frame pages
    # ------------------------------------------------------------------

    def _problem_pages(
        self,
        pdf: PdfPages,
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
        problem_frames_df: pd.DataFrame,
    ) -> None:
        if problem_frames_df is None or problem_frames_df.empty:
            return

        # Title page
        fig = plt.figure(figsize=self.PAGE_FIGSIZE)
        fig.suptitle("DAMAGE / DOOR DETECTED", fontsize=22, fontweight="bold", color="red")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        wagons = problem_frames_df["wagon_id"].drop_duplicates().tolist()
        for wagon_id in wagons:
            frames = problem_frames_df[problem_frames_df["wagon_id"] == wagon_id]
            chunks = [frames.iloc[i:i + self.PROBLEM_FRAMES_PER_PAGE]
                      for i in range(0, len(frames), self.PROBLEM_FRAMES_PER_PAGE)]
            for chunk in chunks:
                self._draw_problem_chunk(pdf, wagon_id, chunk)

    def _draw_problem_chunk(self, pdf, wagon_id, chunk: pd.DataFrame) -> None:
        rows = len(chunk)
        fig, axes = plt.subplots(rows, 1, figsize=self.PAGE_FIGSIZE)
        if rows == 1:
            axes = [axes]
        fig.suptitle(f"Wagon {wagon_id} — Problem Frames", fontsize=14, fontweight="bold")
        for ax, (_, row) in zip(axes, chunk.iterrows()):
            path = row.get("annotated_image_path") or row.get("frame_path")
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"{row['problem_type']} | frame {row['frame_number']}",
                fontsize=10,
            )
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # locomotive pages
    # ------------------------------------------------------------------

    def _loco_pages(self, pdf, loco_summary_df: pd.DataFrame) -> None:
        if loco_summary_df is None or loco_summary_df.empty:
            return

        # loco_summary_df now carries one row per (loco, position) — pick a
        # single representative row per loco (prefer "mid1", matching the old
        # middle-frame behaviour) so the PDF still shows one tile per loco.
        if "position" in loco_summary_df.columns:
            picked_rows = []
            for _, grp in loco_summary_df.groupby("loco_id", sort=True):
                preferred = grp[grp["position"] == "mid1"]
                picked_rows.append(
                    preferred.iloc[0] if not preferred.empty else grp.iloc[len(grp) // 2]
                )
            display_df = pd.DataFrame(picked_rows)
        else:
            display_df = loco_summary_df

        fig = plt.figure(figsize=self.PAGE_FIGSIZE)
        fig.suptitle("LOCOMOTIVES", fontsize=18, fontweight="bold")
        n = len(display_df)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        for i, (_, row) in enumerate(display_df.iterrows(), start=1):
            ax = fig.add_subplot(rows, cols, i)
            path = row.get("frame_path")
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Loco #{int(row['loco_id'])}", fontsize=10)
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # per-wagon pages
    # ------------------------------------------------------------------

    def _wagon_pages(
        self,
        pdf,
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
        problem_frames_df: pd.DataFrame,
    ) -> None:
        if segment_summary_df.empty:
            return
        damage_lookup = (
            damage_results_df.set_index("wagon_id").to_dict("index")
            if not damage_results_df.empty else {}
        )
        problem_lookup: dict[int, pd.DataFrame] = {}
        if problem_frames_df is not None and not problem_frames_df.empty:
            problem_lookup = {
                int(wid): df.head(self.PROBLEM_FRAMES_PER_PAGE)
                for wid, df in problem_frames_df.groupby("wagon_id")
            }

        for _, seg in segment_summary_df.iterrows():
            seg_id = int(seg["segment_id"])
            seg_type = seg.get("segment_type", "wagon")
            start = int(seg["start_frame"])
            end = int(seg["end_frame"])
            seg_dir = seg["directory"]

            fig = plt.figure(figsize=self.PAGE_FIGSIZE)
            damage_row = damage_lookup.get(seg_id, {})
            status, color = self._wagon_status_row(seg_id, damage_row)
            fig.suptitle(
                f"{seg_type.upper()} #{seg_id} — {status}",
                fontsize=14, fontweight="bold", color=color,
            )

            if seg_id in problem_lookup:
                rows = problem_lookup[seg_id]
                for i, (_, row) in enumerate(rows.iterrows(), start=1):
                    ax = fig.add_subplot(1, len(rows), i)
                    path = row.get("annotated_image_path") or row.get("frame_path")
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{row['problem_type']} f{row['frame_number']}", fontsize=9)
                    ax.axis("off")
            else:
                for i, pos in enumerate(self.positions, start=1):
                    frame_idx = int(start + (end - start) * pos)
                    path = os.path.join(seg_dir, f"frame_{frame_idx:06d}.jpg")
                    ax = fig.add_subplot(1, len(self.positions), i)
                    if os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"pos={int(pos * 100)}%", fontsize=9)
                    ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
