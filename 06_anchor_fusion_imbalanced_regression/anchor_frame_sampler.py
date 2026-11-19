# ============================================================
# ANCHOR-BASED FRAME SAMPLER
# ------------------------------------------------------------
# Replaces the old area-signal-guessing sampler. We already KNOW
# the ED and ES frame indices (ground truth for train/val, your
# adaptive_phase_selection predictions for test) so this is a
# lookup + deterministic window build, not a live estimate.
#
# Given (total_frames, ed_frame, es_frame), returns:
#   - a sorted (chronological) array of `num_frames` frame indices
#   - the position of the ED frame WITHIN that sorted array
#   - the position of the ES frame WITHIN that sorted array
#
# The two positions are what the model uses to know exactly which
# of the 32 timesteps should receive the contour+mask injection.
# ============================================================

import numpy as np


def _window_around(center, half_width, total_frames):
    lo = max(0, center - half_width)
    hi = min(total_frames - 1, center + half_width)
    return list(range(lo, hi + 1))


def build_anchor_sequence(total_frames, ed_frame, es_frame,
                           num_frames=32, ed_window=4, es_window=4):
    """
    total_frames: int, total frames in the source video.
    ed_frame, es_frame: int, the KNOWN end-diastolic / end-systolic
                         frame indices (GT for train/val, predicted
                         adaptive-selection for test).
    num_frames: total frame budget (32).
    ed_window / es_window: how many consecutive frames to densely
                            include around each anchor (4 -> 5 frames
                            total per anchor: center +/- 2).

    Returns:
        frame_indices: np.int64 array, length num_frames, sorted ascending.
        ed_position:   int, index into frame_indices where ed_frame landed.
        es_position:   int, index into frame_indices where es_frame landed.
    """
    ed_frame = int(np.clip(ed_frame, 0, total_frames - 1))
    es_frame = int(np.clip(es_frame, 0, total_frames - 1))

    dense_ed = set(_window_around(ed_frame, ed_window // 2, total_frames))
    dense_es = set(_window_around(es_frame, es_window // 2, total_frames))
    selected = dense_ed | dense_es

    remaining_budget = num_frames - len(selected)
    if remaining_budget > 0:
        candidates = [i for i in range(total_frames) if i not in selected]
        if len(candidates) > 0:
            fill_positions = np.linspace(0, len(candidates) - 1, remaining_budget).astype(int)
            fill_positions = np.clip(fill_positions, 0, len(candidates) - 1)
            for pos in fill_positions:
                selected.add(candidates[pos])

    selected = sorted(selected)

    # ---- trim, but NEVER drop the two anchors themselves ----
    if len(selected) > num_frames:
        protected = {ed_frame, es_frame}
        trimmable = [f for f in selected if f not in protected]
        n_to_remove = len(selected) - num_frames
        # remove evenly from the trimmable (filler) frames only
        remove_positions = np.linspace(0, len(trimmable) - 1, n_to_remove).astype(int)
        remove_positions = np.clip(remove_positions, 0, len(trimmable) - 1)
        to_remove = set(trimmable[p] for p in remove_positions)
        selected = sorted(f for f in selected if f not in to_remove)

    # ---- pad if the video is too short to reach num_frames ----
    if len(selected) < num_frames:
        pad_val = selected[-1] if selected else 0
        selected = selected + [pad_val] * (num_frames - len(selected))

    frame_indices = np.array(selected, dtype=np.int64)

    ed_position = int(np.argmin(np.abs(frame_indices - ed_frame)))
    es_position = int(np.argmin(np.abs(frame_indices - es_frame)))

    return frame_indices, ed_position, es_position


if __name__ == "__main__":
    idx, ed_pos, es_pos = build_anchor_sequence(
        total_frames=150, ed_frame=20, es_frame=65, num_frames=32
    )
    print("Selected frame indices:", idx)
    print("Count:", len(idx), "| Unique:", len(set(idx.tolist())))
    print("Is sorted:", list(idx) == sorted(idx))
    print(f"ED (frame {idx[ed_pos]}) landed at sequence position {ed_pos}")
    print(f"ES (frame {idx[es_pos]}) landed at sequence position {es_pos}")
