"""
AutoPan Node — v12  (Raspberry Pi 4B/5 — Blueprint-Aligned)
============================================================
Builds on v9 with geometry fix and full Tier-3 functional upgrades:

  [T1-FIX-1] build_pickup_trajectory / build_deposit_trajectory rewritten
      · All references to deleted ArmParams attributes (shoulder_x, shoulder_z,
        l1, box_x, box_z) replaced with correct v7+ attributes (d1, basket_*).
      · cartesian_interpolate upgraded to 3-D (x, y, z).

  [T1-FIX-2] ik_2dof() calls eliminated — now calls ik_4dof(x, y, z)
      · build_pickup_trajectory, build_deposit_trajectory, run_ik_test all
        updated.  All trajectory waypoints are full 4-tuples (q1,q2,q3,q4).

  [T1-FIX-3] ArtemisLink.send() now unpacks 4-tuple correctly
      · sh, el, wr = angles  →  q1, q2, q3, q4 = angles  (ValueError fixed).

  [T1-FIX-4] Serial packet now sends all 4 joint angles
      · Old packet omitted base rotation (q1) → base servo never moved.
      · New packet: LeftSpeed,RightSpeed,Q1,Q2,Q3,Q4,Gripper\\n
        (Arduino executes directly; no logic required on Arduino side).

  [T1-FIX-5] run_ik_test() now calls ik_4dof — no more NameError.

  [T2-FIX-6] Wheel speeds computed on Pi (P-controller), sent as PWM
      · drive_base_speed, drive_kp added to Config.
      · APPROACH/LOCK: forward + proportional steer on err_x.
      · SEARCH/ALIGN: spin-steer only.  All arm states: stop (0, 0).

  [T2-FIX-7] Ball Y-offset (lateral) saved at LOCK → PICKUP transition
      · ball_y_cm = (err_x_px × dist_ema) / focal_length_px (pinhole).
      · pickup_ball_y stored in FSMContext; used by ik_4dof for base angle.

  [T2-FIX-8] ONNX/TFLite thread count uses os.cpu_count() (Pi 5 aware)
      · Comment corrected; dynamic core detection replaces hard-coded 4.

  [T2-FIX-9] Pickup retry gate on IK failure
      · If >30 % of pickup trajectory waypoints are None, arm_traj is cleared
        and FSM falls back to APPROACH rather than hanging silently.

  [T2-FIX-10] New LIFT state between PICKUP and DEPOSIT
      · Raises arm to shoulder-height + 6 cm before base rotates toward basket.
      · Prevents dragging ball or hitting rover frame during base rotation.

  [GEOM-FIX] basket_y = 2.0 cm (was 0.0) in ArmParams
      · basket exactly on arm-base axis → q1 = atan2(0,-1.7) = 180° which hit
        the q1_max-2° joint-limit margin → every DEPOSIT waypoint was None.
      · 2 cm off-axis → q1 ≈ 174°, well inside limits; arm now moves.

  [T3-FIX-1] Derivative steering — PD controller on err_x (was P only)
      · drive_kd = 0.02 added to Config; ArtemisLink tracks prev_err_x and
        last_send_t internally; steer += kd × (Δerr_x / Δt).
      · Damps oscillation when the rover overshoots centreline at close range.
      · Derivative is reset to 0 when motors are idle (arm states) to prevent
        a stale derivative spike on the next movement.

  [T3-FIX-2] Ball-in-gripper verification after grasp
      · After grip_hold_s elapses in PICKUP, the FSM checks whether a ball
        detection is still visible within dist_lock_cm × 1.5 of the rover.
      · If so → grasp likely failed → gripper_closed reset to False →
        transition back to APPROACH for a fresh run-up.
      · Prevents the arm from lifting air and depositing nothing in the basket.

  [T3-FIX-3] Multi-ball target count
      · Config.target_count (default 3) controls how many balls must be
        collected before the FSM transitions to SUCCESS.
      · VERIFY handler now: collected >= target_count → SUCCESS; else search
        for more balls (drops the old "collected > 0" shortcut).
      · --target-count CLI flag; HUD COLL row shows "n/N"; SUCCESS banner
        shows "n/N balls collected!".

Run modes
---------
    python autopan_v12.py                   # normal
    python autopan_v12.py --no-serial       # vision only
    python autopan_v12.py --port /dev/ttyACM0
    python autopan_v12.py --imgsz 256       # slower Pi
    python autopan_v12.py --debug-hsv
    python autopan_v12.py --calibrate       # MUST re-run on Pi 5 with Pi camera
    python autopan_v12.py --int8            # use INT8 ONNX model
    python autopan_v12.py --target-count 5  # collect 5 balls

Step 1 (once):  python export_onnx.py
Step 2:         python autopan_v12.py

─────── Changelog ────────────────────────────────────────────────────────────
v8   hsv_bands sat floors; hsv_solo_min_conf gate; resolution mismatch WARNING.
v9   T1 crash fixes (trajectory attrs, ik_4dof, 4-tuple unpack, Q1 in packet);
     T2 upgrades (Pi-side PWM, ball_y for IK, LIFT state, IK retry gate,
     dynamic thread count).
v10  Geometry fix (basket_y=2.0); T3 upgrades (PD steering, grasp verify,
     multi-ball target count).
v11  Systematic audit fixes (17 items): DEPOSIT timer bug, IK abort gates,
     TFLite area filter, collect_timeout wired, PD cold-start, downscale
     CFG sync, strong_hsv fast-path, best_dist_cm grasp verify, dead code
     purge, stale doc fixes.
v12  Second audit pass + pre-flight logic fixes (17 items):
     [RED-1]  PICKUP gripper-close timer fixed: same infinite-loop as the v11
              DEPOSIT bug. Added pickup_grip_at field; grip wait now measured
              against it, not against arm_step_at which is reset every 0.8 s.
     [RED-2]  PICKUP / LIFT / DEPOSIT timeout branches now reset gripper_closed
              to False before transitioning to APPROACH — prevents the rover
              from driving around with a permanently closed claw.
     [RED-3]  Grasp verification moved from PICKUP to LIFT: while the arm is
              still on the ground YOLO sees the ball sitting in the open claw
              and falsely flags every successful grasp as a failure. Checking
              after the arm has risen clears the camera view of the ball.
     [RED-4]  LIFT / DEPOSIT IK abort and timeout branches also reset
              gripper_closed to False (were missing this in v11).
     [ORG-5]  _PTEngine (PyTorch fallback) max_box_area filter added — the
              v11 fix covered ONNX and TFLite but missed the third engine.
     [ORG-6]  Kalman now smooths best_dist_cm (fused bbox+ellipse) instead of
              raw dist_cm (bbox-only). ellipse_b zeroed post-replacement so
              best_dist_cm doesn't double-count the now-absorbed ellipse delta.
     [ORG-7]  Kalman jump-reset threshold widened 60 px → 120 px. At ≤15 cm
              the ball fills most of the frame; the 60 px gate fired on every
              legitimate frame and kept resetting the filter mid-approach.
     [ORG-8]  SEARCH cold-start spin fixed: FSMContext.cx initialised to
              cam_w/2 after construction so err_x=0 on first frame instead of
              err_x=-320 → hard right-spin before any ball is seen.
     [ORG-9]  LOCK → PICKUP now saves ctx.dist (Kalman-filtered best_dist_cm)
              as pickup_ball_x instead of ctx.dist_ema (alpha=0.15, still
              dominated by approach-phase readings at transition time).
     [YLW-10] PICKUP on_enter hook: set-then-clear arm_traj pattern replaced
              with consistent set-in-else pattern matching LIFT/DEPOSIT hooks.
     [YLW-11] ALIGN on_enter log: "err=%.0fpx" now shows c.cx - cam_w/2
              (actual pixel error from centre) not raw c.cx (column position).
     [YLW-12] HSV mask comment "5 bands bitwise-OR'd" corrected to "3 bands".
     [YLW-13] Duplicate HSV band comment block in Config removed.
     [YLW-14] `import sys` moved to module level; all three local `import sys`
              calls (CameraStream.__init__, FPSWatchdog.update) removed.
     [YLW-15] `import cv2/numpy/time` inside CameraStream warm-up loop removed
              — module-level imports already available; loop is now clean.
     [BLU-16] run_calibration() sets CAP_PROP_FRAME_WIDTH/HEIGHT to CFG values
              so the measured focal_length_px is valid at the runtime resolution.
     [BLU-17] Gradient weight 0.08 extracted to Config.w_grad so the detection
              confidence formula is fully tunable without editing code.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import signal
import sys          # [v12] moved from local imports inside CameraStream / FPSWatchdog
import threading
import time
import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

try:
    import onnxruntime as ort
    _ONNX_OK = True
except ImportError:
    _ONNX_OK = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    # Try the lightweight Raspberry Pi package first
    import tflite_runtime.interpreter as tflite
    _TFLITE_OK = True
except ImportError:
    try:
        # Fallback to full TensorFlow for desktop testing
        import tensorflow as tf
        tflite = tf.lite
        _TFLITE_OK = True
    except ImportError:
        _TFLITE_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AutoPan")


# ══════════════════════════════════════════════════════════════════════════════
# Arm Kinematics  — Full 4-DOF Geometric IK (blueprint-exact)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ArmParams:
    """
    Physical dimensions from hardware spec (all in cm).

    d1  = Base height from ground to shoulder pivot  = 150 mm → 15.0 cm
    L2  = Shoulder link (upper arm)                  = 135 mm → 13.5 cm
    L3  = Elbow link (forearm)                       = 120 mm → 12.0 cm
    L4  = Wrist link                                 =  83 mm →  8.3 cm
    Total reach ≈ L2+L3+L4 = 33.8 cm from shoulder pivot

    Basket position in rover frame (behind arm, on rover back):
        X = -100 mm → -10.0 cm
        Y =   20 mm →   2.0 cm  [GEOM-FIX] off-axis keeps q1 away from ±180° limit
        Z =  200 mm →  20.0 cm
    """
    d1:  float = 15.0   # base / column height (cm)
    L2:  float = 13.5   # shoulder link (cm)
    L3:  float = 12.0   # elbow link (cm)
    L4:  float =  8.3   # wrist link (cm)

    # Joint limits in DEGREES (physical hardware limits)
    q1_min: float = -180.0;  q1_max: float =  180.0   # base (full rotation)
    q2_min: float =  -30.0;  q2_max: float =  160.0   # shoulder
    q3_min: float =    0.0;  q3_max: float =  170.0   # elbow
    q4_min: float =  -90.0;  q4_max: float =   90.0   # wrist

    # Basket deposit position (rover-frame, cm)
    basket_x: float = -10.0
    basket_y: float =   2.0  # [GEOM-FIX] 2 cm off-axis: basket_y=0 → q1=180°
                              # which hit joint-limit margin → all DEPOSIT IK = None
    basket_z: float =  20.0  # = 200 mm above ground

    # Pickup target height (cm above ground) — centre of a 67 mm tennis ball
    pickup_z_cm: float = 3.35   # ≈ half of 67 mm ball diameter


ARM = ArmParams()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ik_4dof(
    target_x: float,
    target_y: float,
    target_z: float,
    desired_orientation_deg: float = 0.0,
    elbow_up: bool = True,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Full 4-DOF geometric (closed-form) Inverse Kinematics.

    All inputs in centimetres, in the rover body frame:
        +X = forward,  +Y = left,  +Z = up

    Returns (q1, q2, q3, q4) in degrees, or None if unreachable.

    Derivation (blueprint-exact):
      · The end-effector must reach (target_x, target_y, target_z).
      · The wrist pivot is L4 shorter than the end-effector along the
        horizontal plane (we subtract the wrist link contribution first).
      · r  = horizontal distance from base axis to wrist pivot
        s  = (target_z - d1) — vertical rise from shoulder pivot
      · q1 = atan2(y, x)                               [base rotation]
      · q3 = acos((r²+s²-L2²-L3²) / (2·L2·L3))        [elbow]
      · q2 = atan2(s,r) - atan2(L3·sin(q3), L2+L3·cos(q3))  [shoulder]
      · q4 = desired_orientation - (q2+q3)              [wrist, keeps gripper level]
    """
    # Wrist pivot position (end-effector minus wrist link along XY plane)
    horiz_dist = math.sqrt(target_x**2 + target_y**2)
    # If horizontal distance is zero (directly above base), point forward
    if horiz_dist < 1e-6:
        wx, wy = target_x, target_y
    else:
        scale = (horiz_dist - ARM.L4) / horiz_dist
        wx = target_x * scale
        wy = target_y * scale
    wz = target_z   # wrist z = target z (wrist link is horizontal)

    # ── q1: base rotation ────────────────────────────────────────────────────
    q1_rad = math.atan2(wy, wx)
    q1_deg = math.degrees(q1_rad)

    # ── Planar IK for shoulder + elbow in the sagittal plane ─────────────────
    # r = horizontal distance to wrist pivot;  s = vertical rise from shoulder
    r = math.sqrt(wx**2 + wy**2)
    s = wz - ARM.d1   # height above shoulder pivot

    r2  = r**2 + s**2
    den = 2.0 * ARM.L2 * ARM.L3

    # Reachability
    max_r = ARM.L2 + ARM.L3
    min_r = abs(ARM.L2 - ARM.L3)
    reach = math.sqrt(r2)
    if reach > max_r * 0.97 or reach < min_r * 1.03:
        log.warning("IK: target (%.1f,%.1f,%.1f) unreachable (r=%.1f, max=%.1f)",
                    target_x, target_y, target_z, reach, max_r)
        return None

    # q3 — elbow (law of cosines)
    cos_q3 = _clamp((r2 - ARM.L2**2 - ARM.L3**2) / den, -1.0, 1.0)
    sin_q3 = math.sqrt(max(0.0, 1.0 - cos_q3**2))
    if not elbow_up:
        sin_q3 = -sin_q3
    q3_rad = math.atan2(sin_q3, cos_q3)

    # q2 — shoulder
    gamma   = math.atan2(s, r)
    alpha   = math.atan2(ARM.L3 * sin_q3, ARM.L2 + ARM.L3 * cos_q3)
    q2_rad  = gamma - alpha

    # q4 — wrist (keeps gripper level with ground → total pitch = 0)
    q2_deg = math.degrees(q2_rad)
    q3_deg = math.degrees(q3_rad)
    q4_deg = desired_orientation_deg - (q2_deg + q3_deg)

    # ── Joint-limit check with 2° margin ─────────────────────────────────────
    margin = 2.0
    checks = [
        (q1_deg, ARM.q1_min, ARM.q1_max, "q1 base"),
        (q2_deg, ARM.q2_min, ARM.q2_max, "q2 shoulder"),
        (q3_deg, ARM.q3_min, ARM.q3_max, "q3 elbow"),
    ]
    for val, lo, hi, name in checks:
        if not (lo + margin <= val <= hi - margin):
            log.warning("IK: %s=%.1f° out of limits [%.0f, %.0f]", name, val, lo, hi)
            return None

    q4_deg = _clamp(q4_deg, ARM.q4_min, ARM.q4_max)

    return q1_deg, q2_deg, q3_deg, q4_deg


def cartesian_interpolate(
    start: Tuple[float, float, float],
    end:   Tuple[float, float, float],
    steps: int = 10,
) -> List[Tuple[float, float, float]]:
    """
    Linear Cartesian interpolation in (x, y, z) space.
    Returns `steps` waypoints INCLUDING start and end.
    Allows smooth straight-line end-effector motion.
    steps=1 returns just the end point.
    """
    if steps < 2:
        return [end]
    waypoints = []
    for i in range(steps):
        t = i / (steps - 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        z = start[2] + t * (end[2] - start[2])
        waypoints.append((x, y, z))
    return waypoints


def build_pickup_trajectory(
    ball_x_cm: float,
    ball_y_cm: float = 0.0,
    steps: int = 10,
) -> List[Optional[Tuple[float, float, float, float]]]:
    """
    Generate a sequence of (q1, q2, q3, q4) joint angles for a smooth PICKUP
    motion using full 4-DOF IK:
        Park → Hover (5 cm above ball) → Grasp

    ball_x_cm : forward distance to ball in rover frame (cm)
    ball_y_cm : lateral offset  (positive = left of rover centreline, cm)
                derived from pixel error at LOCK: y = err_x_px * dist / focal

    Returns a list of 4-tuple angles.  None entries mean IK failure for that
    waypoint; if >30 % are None the caller should abort (see PICKUP update).
    """
    # Park = safe retracted position near basket (arm won't drag on rotate)
    park  = (ARM.basket_x, ARM.basket_y, ARM.basket_z + 5.0)
    hover = (ball_x_cm, ball_y_cm, ARM.pickup_z_cm + 5.0)   # 5 cm above ball
    grasp = (ball_x_cm, ball_y_cm, ARM.pickup_z_cm)

    trajectory: List[Optional[Tuple[float, float, float, float]]] = []
    prev = park
    for target, n in ((hover, steps), (grasp, max(steps // 2, 2))):
        for pt in cartesian_interpolate(prev, target, n):
            trajectory.append(ik_4dof(*pt))
        prev = target
    return trajectory


def build_lift_trajectory(
    ball_x_cm: float,
    ball_y_cm: float = 0.0,
    steps: int = 6,
) -> List[Optional[Tuple[float, float, float, float]]]:
    """
    Raise the arm straight up from pickup height to a safe clearance height
    (ARM.d1 + 6 cm) before the base servo rotates toward the basket.
    Prevents ball drag and rover-frame collisions during base rotation.
    """
    start = (ball_x_cm, ball_y_cm, ARM.pickup_z_cm)
    end   = (ball_x_cm, ball_y_cm, ARM.d1 + 6.0)   # 21 cm from ground — clears frame

    trajectory: List[Optional[Tuple[float, float, float, float]]] = []
    for pt in cartesian_interpolate(start, end, steps):
        trajectory.append(ik_4dof(*pt))
    return trajectory


def build_deposit_trajectory(
    ball_x_cm: float = 0.0,
    ball_y_cm: float = 0.0,
    steps: int = 10,
) -> List[Optional[Tuple[float, float, float, float]]]:
    """
    Move arm from lifted clearance position to basket deposit position.
    Called after LIFT completes, so the arm starts at (ball_x, ball_y, d1+6).
    """
    # Start from the lifted clearance position (matches build_lift_trajectory end)
    lifted  = (ball_x_cm, ball_y_cm, ARM.d1 + 6.0)
    deposit = (ARM.basket_x, ARM.basket_y, ARM.basket_z)

    trajectory: List[Optional[Tuple[float, float, float, float]]] = []
    for pt in cartesian_interpolate(lifted, deposit, steps):
        trajectory.append(ik_4dof(*pt))
    return trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Kalman Filter  (replaces EMA — handles measurement gaps cleanly)
# ══════════════════════════════════════════════════════════════════════════════
class KalmanFilter1D:
    """
    Constant-velocity 1-D Kalman for a single measurement stream.
    State = [position, velocity].
    """
    def __init__(self, process_noise: float = 1.0, meas_noise: float = 5.0):
        self._x  = np.zeros(2)               # [pos, vel]
        self._P  = np.eye(2) * 100.0         # covariance
        self._Q  = np.eye(2) * process_noise # process noise
        self._R  = np.array([[meas_noise]])   # measurement noise
        self._F  = np.eye(2)                 # transition (updated each step)
        self._H  = np.array([[1.0, 0.0]])    # only measure position
        self._init = False

    def update(self, z: float, dt: float = 0.033) -> float:
        if not self._init:
            self._x[0] = z; self._init = True; return z

        self._F[0, 1] = dt   # position += velocity * dt

        # Predict
        x_p = self._F @ self._x
        P_p = self._F @ self._P @ self._F.T + self._Q

        # Update
        y   = np.array([z]) - self._H @ x_p          # shape (1,)
        S   = self._H @ P_p @ self._H.T + self._R     # shape (1,1)
        K   = (P_p @ self._H.T / S[0, 0]).ravel()    # shape (2,)  ← squeeze fixes the crash
        self._x = x_p + K * float(y[0])              # (2,) + (2,) → (2,)
        self._P = (np.eye(2) - np.outer(K, self._H)) @ P_p

        return float(self._x[0])

    def reset(self): self._init = False; self._P = np.eye(2) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Camera
    cam_src: int   = 0
    cam_w:   int   = 640
    cam_h:   int   = 480

    # Model
    onnx_path:     str   = "yolov8n.onnx"
    onnx_int8_path: str  = "yolov8n_int8.onnx"   # quantised — fastest on Pi
    pt_path:       str   = "yolov8n.pt"
    infer_imgsz:   int   = 320
    infer_conf:    float = 0.30    # lowered: catch partial/distant balls
    infer_iou:     float = 0.35
    yolo_class:    int   = 32      # COCO "sports ball"
    tflite_path:   str   = "yolov8n_saved_model/yolov8n_int8.tflite"
    # ── HSV bands — tennis ball only ─────────────────────────────────────────
    # Tennis felt hue = 28-42 (lime-yellow).
    # Band 1: primary lit face   — high saturation, lime-yellow core
    # Band 2: shadow face        — sat floor raised 60→90: was leaking cream/beige
    #                              under warm indoor lighting (pool balls, walls)
    # Band 3: worn/dirty ball    — sat floor raised 50→80: prevents glare patches
    #                              and off-white surfaces from matching
    # Raising the saturation floors is the key change: desaturated colours
    # (cream, white-grey, skin) have sat<80 in HSV.  Tennis felt's desaturated
    # "worn" face still sits comfortably above sat=80 (measured ~90-110).
    hsv_bands: tuple = (
        ((28, 120, 100), (42, 255, 255)),   # 1. primary lit face  (unchanged)
        ((27,  90,  70), (43, 180, 220)),   # 2. shadow face       (sat floor 60→90)
        ((26,  80,  60), (44, 150, 200)),   # 3. worn / dirty      (sat floor 50→80)
    )
    hsv_ratio_min:   float = 0.22    # [v6] relaxed for small distant blobs
    hsv_fast_thresh: float = 0.55    # only skip YOLO for confident hits

    # Shape — strict to keep faces, rectangles, and window strips out
    circularity_min:  float = 0.55   # [v6] slightly relaxed for distant small blobs
    min_box_area:     int   = 200    # [v6] lowered: ~14×14 px covers 200 cm ball
    max_box_area:     int   = 160000 # [v6.1] raised: ball at 15 cm = ~154k px²
    min_box_px:       int   = 12     # [v6] lowered for 200 cm detection (~29 px wide)
    max_box_px:       int   = 400    # [v6.1] raised: ball at 15 cm = ~392 px wide
    aspect_min:       float = 0.55   # nearly square
    aspect_max:       float = 1.80   # nearly square
    ellipse_fit_min:  int   = 5      # [v6] fewer points needed for tiny blobs
    max_hsv_dets:     int   = 3      # cap HSV candidates before fusion

    # YOLO detections rejected unless HSV mask also covers them
    yolo_min_hsv_ratio: float = 0.12  # [v6] relaxed for distant balls (small bbox)
    # [v7-fix] HSV-only blobs (no YOLO cross-confirmation) are dropped unless their
    # confidence meets this floor.  Keeps genuine far-away balls (fused as "hybrid")
    # while discarding stray pool balls / glare that YOLO correctly ignored.
    hsv_solo_min_conf:  float = 0.45

    # Hough
    use_hough:    bool  = False
    hough_dp:     float = 1.2
    hough_param1: int   = 55
    hough_param2: int   = 18
    hough_min_r:  int   = 6
    hough_max_r:  int   = 130

    # Fusion
    iou_merge_thresh: float = 0.22   # lower → merge more aggressively
    w_yolo:  float = 0.45
    w_hsv:   float = 0.35
    w_circ:  float = 0.20
    w_grad:  float = 0.08   # [v12] gradient smoothness bonus (was hardcoded literal)

    # Distance — now uses both axes if ellipse fitted
    ball_diameter_cm: float = 6.7
    # Logitech Brio 100: 58° DFOV (not ~70° — that was wrong), 1920×1080 native.
    # This value (877.6 px) was empirically measured via --calibrate at 640×480
    # and is camera + resolution specific.  If running on the Pi 5 with a
    # different camera (Pi Camera Module, or even the same Brio but through
    # libcamera/v4l2 at a different crop), the effective focal length in pixels
    # WILL differ → re-run:  python autopan_v11.py --calibrate
    # Theoretical sanity check: diag_px=800, dFoV=58° → f≈721 px.
    # The measured 877.6 is plausible given lens distortion / crop at 640×480.
    focal_length_px:  float = 877.6
    # [v6] confidence decay starts at 120 cm (was 60) to support 200 cm detection
    dist_decay_start_cm:      float = 120.0
    dist_conf_decay_per_10cm: float = 0.025  # [v6] gentler decay (was 0.04)

    # FSM thresholds (cm)
    dist_lock_cm:      float = 15.0
    dist_approach_cm:  float = 25.0
    align_px_thresh:   int   = 25      # centre error before driving
    search_timeout_s:  float = 5.0
    lost_frames_limit: int   = 35      # [v6] increased: tolerate brief occlusions
    lock_hold_frames:  int   = 28      # [v6] increased: more stable lock-to-pickup
    collect_timeout_s: float = 30.0   # [v11] per-arm-state hang guard (was 6 s —
                                      # too short for 15-waypoint pickup @ 0.8 s/step)
    pickup_step_s:     float = 0.8    # time per trajectory step
    deposit_step_s:    float = 0.8
    verify_duration_s: float = 1.8
    grip_hold_s:       float = 0.5   # time gripper stays closed before retract
    target_count:      int   = 3     # [T3] balls to collect before SUCCESS
    # [v6] Detection stability: require N consecutive frames before accepting new target
    det_confirm_frames: int  = 3      # new target must appear in N frames to be valid
    # [v6] Distance EMA for FSM decisions (separate from Kalman used for servo output)
    fsm_dist_ema_alpha: float = 0.15  # lower = more lag but smoother FSM decisions

    # [v6] Kalman process/measurement noise
    # Higher meas_xy / meas_dist → trust model more, reject sensor jitter
    kalman_proc: float = 1.0        # [v6] reduced process noise (smoother prediction)
    kalman_meas_xy: float = 14.0    # [v6] higher: Kalman trusts model over noisy px
    kalman_meas_dist: float = 20.0  # [v6] higher: distance sensor noisier at 2 m

    # Serial — auto-detects OS: COM3 on Windows, /dev/ttyACM0 on Linux/Pi
    serial_port:           str   = "COM3" if __import__("sys").platform == "win32" else "/dev/ttyACM0"
    serial_baud:           int   = 115200
    serial_timeout:        float = 1.0
    serial_reconnect_base: float = 1.0
    serial_reconnect_max:  float = 30.0
    serial_queue_size:     int   = 10
    no_serial:             bool  = False

    # [v9] Wheel speed controller — computed on Pi, sent as PWM to Arduino
    # Arduino simply executes: left motor = LeftSpeed, right motor = RightSpeed
    drive_base_speed: int   = 150    # base forward PWM [0-255]
    drive_kp:         float = 0.5    # proportional steer gain on err_x (px)
    drive_kd:         float = 0.02   # [T3] derivative steer gain — damps overshoot

    # FPS watchdog — auto-reduce resolution if FPS drops
    fps_target:  float = 14.0
    fps_low_s:   float = 5.0    # seconds below target before downsample
    use_int8:    bool  = False   # --int8 flag

    # [v6] Dark-frame gamma correction threshold
    # Frames with mean luminance below this get gamma=2.2 lift before CLAHE+HSV.
    dark_frame_thresh: float = 40.0   # 0-255; typical indoor dim = 15, normal = 80+

    # Debug
    debug_hsv:  bool = False
    infer_skip_n: int = 2   # run YOLO every N frames (1 = every frame)


CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
# Detection dataclass
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    cx: float; cy: float
    width: int; height: int; area: int
    dist_cm: float
    confidence: float
    circularity: float
    source: str
    hsv_ratio: float
    # NEW: ellipse axes (a=semi-major, b=semi-minor) in pixels, 0 if not fitted
    ellipse_a: float = 0.0
    ellipse_b: float = 0.0

    @property
    def bbox(self): return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, o: "Detection") -> float:
        ix1 = max(self.x1, o.x1); iy1 = max(self.y1, o.y1)
        ix2 = min(self.x2, o.x2); iy2 = min(self.y2, o.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if not inter: return 0.0
        return inter / (self.area + o.area - inter)

    @property
    def ellipse_dist_cm(self) -> float:
        """
        Distance estimate from the MINOR axis of the fitted ellipse.
        For a ball seen at an angle, the minor axis is the true circle
        projected onto the image plane — more accurate than bbox width.
        """
        if self.ellipse_b > 0:
            return (CFG.ball_diameter_cm * CFG.focal_length_px) / (2 * self.ellipse_b)
        return self.dist_cm

    @property
    def best_dist_cm(self) -> float:
        """Average bbox-based and ellipse-based estimates if both available."""
        if self.ellipse_b > 0 and self.dist_cm > 0:
            return 0.5 * (self.dist_cm + self.ellipse_dist_cm)
        return self.dist_cm


# ══════════════════════════════════════════════════════════════════════════════
# ONNX Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
class ONNXEngine:
    def __init__(self, path: str):
        opts = ort.SessionOptions()
        n_cores = os.cpu_count() or 4   # Pi 4B=4 × A72; Pi 5=4 × A76 (~2-3× faster)
        opts.intra_op_num_threads  = n_cores
        opts.inter_op_num_threads  = max(1, n_cores // 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Enable ARM NEON SIMD on Pi
        opts.add_session_config_entry("session.enable_mem_arena", "1")
        self._sess = ort.InferenceSession(
            path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._inp_name = self._sess.get_inputs()[0].name
        self._imgsz    = CFG.infer_imgsz
        log.info("ONNX engine loaded: %s  (inp='%s')", path, self._inp_name)
        # Warm up 2 passes to pre-alloc internal buffers
        dummy = np.zeros((1, 3, self._imgsz, self._imgsz), np.float32)
        for _ in range(2):
            self._sess.run(None, {self._inp_name: dummy})
        log.info("ONNX warm-up done.")

    def predict(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)
        raw  = self._sess.run(None, {self._inp_name: blob})[0]  # [1, 84, N]
        raw  = raw[0].T   # → [N, 84]

        class_scores = raw[:, 4:]
        conf  = class_scores[:, CFG.yolo_class]
        mask  = conf >= CFG.infer_conf
        raw_f = raw[mask]; conf_f = conf[mask]

        if len(raw_f) == 0:
            return []

        cx_n = raw_f[:, 0]; cy_n = raw_f[:, 1]
        bw_n = raw_f[:, 2]; bh_n = raw_f[:, 3]
        sx   = orig_w / self._imgsz
        sy   = orig_h / self._imgsz

        x1s  = ((cx_n - bw_n / 2) * sx).astype(int)
        y1s  = ((cy_n - bh_n / 2) * sy).astype(int)
        x2s  = ((cx_n + bw_n / 2) * sx).astype(int)
        y2s  = ((cy_n + bh_n / 2) * sy).astype(int)

        dets = []
        for x1, y1, x2, y2, c in zip(x1s, y1s, x2s, y2s, conf_f):
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(orig_w, x2); y2 = min(orig_h, y2)
            w  = x2 - x1; h = y2 - y1; area = w * h
            if area < CFG.min_box_area: continue
            if area > CFG.max_box_area: continue   # [v6.1] consistent with HSV path
            dist = (CFG.ball_diameter_cm * CFG.focal_length_px) / w if w > 0 else 0.0
            dets.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                width=w, height=h, area=area,
                dist_cm=dist, confidence=float(c),
                circularity=0.0, source="yolo", hsv_ratio=0.0,
            ))
        return self._nms(dets)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        s   = self._imgsz
        img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None]  # NCHW

    @staticmethod
    def _nms(dets: List[Detection]) -> List[Detection]:
        if not dets: return []
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept = []
        for d in dets:
            if all(d.iou(k) < CFG.infer_iou for k in kept):
                kept.append(d)
        return kept

# ══════════════════════════════════════════════════════════════════════════════
# TFLITE Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
class TFLiteEngine:
    def __init__(self, path: str):
        # Load the TFLite model and allocate tensors
        self._interpreter = tflite.Interpreter(model_path=path, num_threads=os.cpu_count() or 4)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._input_idx = self._input_details[0]['index']
        self._output_idx = self._output_details[0]['index']

        # Check if the model expects NCHW [1, 3, imgsz, imgsz] or NHWC [1, imgsz, imgsz, 3]
        input_shape = self._input_details[0]['shape']
        self._is_nhwc = input_shape[-1] == 3
        self._imgsz = CFG.infer_imgsz

        log.info("TFLite engine loaded: %s", path)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]

        # --- 1. Preprocess ---
        s = self._imgsz
        img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Format layout
        if self._is_nhwc:
            blob = img[None]  # NHWC
        else:
            blob = img.transpose(2, 0, 1)[None]  # NCHW

        # Handle INT8 quantized input
        if self._input_details[0]['dtype'] == np.int8:
            scale, zero_point = self._input_details[0]['quantization']
            blob = (blob / scale + zero_point).astype(np.int8)

        # --- 2. Infer ---
        self._interpreter.set_tensor(self._input_idx, blob)
        self._interpreter.invoke()

        # --- 3. Postprocess ---
        raw = self._interpreter.get_tensor(self._output_idx)[0]

        # De-quantize output if it's INT8
        if self._output_details[0]['dtype'] == np.int8:
            scale, zero_point = self._output_details[0]['quantization']
            raw = (raw.astype(np.float32) - zero_point) * scale

        # Ensure correct shape [N, 84]
        if raw.shape[0] == 84:
            raw = raw.T

        class_scores = raw[:, 4:]
        conf = class_scores[:, CFG.yolo_class]
        mask = conf >= CFG.infer_conf
        raw_f = raw[mask];
        conf_f = conf[mask]

        if len(raw_f) == 0:
            return []

        cx_n = raw_f[:, 0];
        cy_n = raw_f[:, 1]
        bw_n = raw_f[:, 2];
        bh_n = raw_f[:, 3]
        sx = orig_w / self._imgsz
        sy = orig_h / self._imgsz

        x1s = ((cx_n - bw_n / 2) * sx).astype(int)
        y1s = ((cy_n - bh_n / 2) * sy).astype(int)
        x2s = ((cx_n + bw_n / 2) * sx).astype(int)
        y2s = ((cy_n + bh_n / 2) * sy).astype(int)

        dets = []
        for x1, y1, x2, y2, c in zip(x1s, y1s, x2s, y2s, conf_f):
            x1 = max(0, x1);
            y1 = max(0, y1)
            x2 = min(orig_w, x2);
            y2 = min(orig_h, y2)
            w = x2 - x1;
            h = y2 - y1;
            area = w * h
            if area < CFG.min_box_area: continue
            if area > CFG.max_box_area: continue   # [v11] close-range giant blob guard — was missing in TFLite path
            dist = (CFG.ball_diameter_cm * CFG.focal_length_px) / w if w > 0 else 0.0
            dets.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                width=w, height=h, area=area,
                dist_cm=dist, confidence=float(c),
                circularity=0.0, source="yolo", hsv_ratio=0.0,
            ))
        return self._nms(dets)

    @staticmethod
    def _nms(dets: List[Detection]) -> List[Detection]:
        if not dets: return []
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept = []
        for d in dets:
            if all(d.iou(k) < CFG.infer_iou for k in kept):
                kept.append(d)
        return kept
# ══════════════════════════════════════════════════════════════════════════════
# Async Inference Worker
# ══════════════════════════════════════════════════════════════════════════════
class AsyncInferenceWorker:
    def __init__(self, engine):
        self._engine  = engine
        self._in_q:  queue.Queue = queue.Queue(maxsize=2)
        self._out_q: queue.Queue = queue.Queue(maxsize=2)
        self._latest: List[Detection] = []
        self._run = True
        threading.Thread(target=self._loop, daemon=True, name="Infer").start()

    def submit(self, frame: np.ndarray):
        try:
            self._in_q.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_q.put_nowait(frame.copy())
            except queue.Full:
                pass

    def get_latest(self) -> List[Detection]:
        try:
            while True:
                self._latest = self._out_q.get_nowait()
        except queue.Empty:
            pass
        return self._latest

    def stop(self): self._run = False

    def _loop(self):
        while self._run:
            try:
                frame  = self._in_q.get(timeout=0.5)
                result = self._engine.predict(frame)
                try:
                    self._out_q.put_nowait(result)
                except queue.Full:
                    try: self._out_q.get_nowait()
                    except queue.Empty: pass
                    try: self._out_q.put_nowait(result)
                    except queue.Full: pass
            except queue.Empty:
                pass
            except Exception as e:
                log.error("Inference error: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# Ball Detector  — HSV + ellipse + YOLO fusion
# ══════════════════════════════════════════════════════════════════════════════
class BallDetector:
    """
    3-stage pipeline:
      1. CLAHE normalisation
      2. HSV mask → contour → ellipse fitting → geometric confidence
      3. YOLO async → enrich with HSV/ellipse → fuse with NMS boost
    """

    def __init__(self, worker: AsyncInferenceWorker):
        self._worker = worker
        # Pre-alloc morph kernels
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Pre-compile HSV band arrays
        self._hsv_bands = [
            (np.array(lo, np.uint8), np.array(hi, np.uint8))
            for lo, hi in CFG.hsv_bands
        ]
        # CLAHE instance (cached — avoid per-frame alloc)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._frame_count = 0
        # [v6] Pre-build gamma LUT for dark-frame correction (gamma=2.2)
        self._gamma_lut = np.array(
            [(i / 255.0) ** (1.0 / 2.2) * 255 for i in range(256)], dtype=np.uint8
        )

    # ── Dark-frame adaptive gamma ─────────────────────────────────────────────
    def _maybe_gamma(self, frame: np.ndarray) -> np.ndarray:
        """
        [v6] If the frame is very dark (camera AE not yet settled, or dim
        environment), apply gamma=2.2 correction BEFORE CLAHE.
        Threshold: mean luminance < 40 out of 255.
        At normal exposure this adds ~0.1 ms and returns the frame unchanged.
        """
        mean_v = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if mean_v < CFG.dark_frame_thresh:
            return cv2.LUT(frame, self._gamma_lut)
        return frame

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._frame_count += 1

        # [v6] Lift dark frames before any colour processing
        frame_adj = self._maybe_gamma(frame)
        preproc   = self._apply_clahe(frame_adj)
        hsv       = cv2.cvtColor(preproc, cv2.COLOR_BGR2HSV)
        mask      = self._hsv_mask(hsv)

        if CFG.debug_hsv:
            cv2.imshow("HSV Mask", mask)

        # Stage 1: HSV + ellipse blobs (cheap, always runs)
        # Pass frame_adj so blob scoring uses the brightness-corrected image
        hsv_dets = self._hsv_blobs(frame_adj, mask)

        # Fast-path: high-confidence HSV + circularity hit → skip YOLO this frame.
        # [v11] Previously computed but YOLO was submitted unconditionally — the
        # list was built and immediately discarded.  Now actually gates the submit.
        strong_hsv = [d for d in hsv_dets
                      if d.hsv_ratio  >= CFG.hsv_fast_thresh
                      and d.circularity >= CFG.circularity_min]

        # Submit brightness-corrected frame to YOLO (skip if strong HSV hit)
        if not strong_hsv and self._frame_count % CFG.infer_skip_n == 0:
            self._worker.submit(preproc)

        yolo_dets = self._worker.get_latest()
        yolo_enriched = [
            e for d in yolo_dets
            if (e := self._enrich(d, frame, mask)) is not None
        ]

        # Fuse
        fused = self._fuse(yolo_enriched, hsv_dets)

        # Apply distance-based confidence decay and final filter
        valid = []
        for d in fused:
            if d.dist_cm > CFG.dist_decay_start_cm:
                n_decades = (d.dist_cm - CFG.dist_decay_start_cm) / 10.0
                d = dataclasses.replace(
                    d, confidence=d.confidence - n_decades * CFG.dist_conf_decay_per_10cm
                )
            if (d.circularity >= CFG.circularity_min
                    and d.hsv_ratio  >= CFG.hsv_ratio_min
                    and d.confidence >= 0.25):
                # [v7-fix] Require YOLO backing for low-confidence colour-only blobs.
                # A real tennis ball almost always gets both HSV AND YOLO confirmation;
                # a pool ball / glare patch / random yellow object typically gets only one.
                # HSV-only (source="hsv") blobs below 0.45 confidence are discarded here
                # because they have no YOLO cross-validation and the tightened HSV bands
                # alone are not sufficient to exclude all non-tennis-ball objects.
                if d.source == "hsv" and d.confidence < CFG.hsv_solo_min_conf:
                    continue
                valid.append(d)

        # Sort nearest-first: the closest confirmed ball is always the primary target.
        valid.sort(key=lambda d: d.best_dist_cm)

        # [v7-fix] Return at most ONE detection.
        # Returning a list of 2-3 objects caused the HUD to show BALL 3 and allowed
        # the FSM / Kalman to lock onto non-tennis-ball objects as secondary targets.
        # The FSM only ever needs dets[0]; discarding the rest here is safe.
        return valid[:1]

    # ── CLAHE (cached instance) ───────────────────────────────────────────────
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    # ── HSV mask — 3 bands bitwise-OR'd ──────────────────────────────────────
    def _hsv_mask(self, hsv: np.ndarray) -> np.ndarray:
        out = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self._hsv_bands:
            out = cv2.bitwise_or(out, cv2.inRange(hsv, lo, hi))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, self._k_close)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  self._k_open)
        return out

    # ── HSV + ellipse blobs ───────────────────────────────────────────────────
    def _hsv_blobs(self, frame: np.ndarray,
                   mask: np.ndarray) -> List[Detection]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # ── Size gates ──────────────────────────────────────────────────
            if area < CFG.min_box_area: continue
            if area > CFG.max_box_area: continue          # blob too large

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Side-length gates (pixels)
            if bw < CFG.min_box_px or bh < CFG.min_box_px: continue
            if bw > CFG.max_box_px or bh > CFG.max_box_px: continue

            # Aspect ratio — tennis ball is nearly square from any angle.
            # [v6.1] Skip aspect gate if blob touches a frame edge: at ≤20 cm
            # the ball may overflow the frame and produce a clipped rectangle
            # whose aspect ratio is meaningless.
            fh, fw = mask.shape[:2]
            edge_clipped = (bx <= 0 or by <= 0 or
                            bx + bw >= fw or by + bh >= fh)
            if not edge_clipped:
                aspect = bw / bh if bh else 0
                if not (CFG.aspect_min < aspect < CFG.aspect_max): continue

            # --- Ellipse fitting (key improvement) ---
            el_a = el_b = 0.0
            ellipse_circ = 0.0
            if len(cnt) >= CFG.ellipse_fit_min:
                try:
                    (ex, ey), (ea, eb), angle = cv2.fitEllipse(cnt)
                    # ea = full width, eb = full height of ellipse
                    el_a = max(ea, eb) / 2   # semi-major
                    el_b = min(ea, eb) / 2   # semi-minor
                    # Ellipse circularity: ratio of minor to major axis
                    if el_a > 0:
                        ellipse_circ = el_b / el_a
                except Exception:
                    pass

            # Contour circularity (shape boundary roundness)
            circ_contour = self._contour_circularity(cnt)

            # Combined circularity: prefer ellipse fit when available
            if el_b > 0:
                circ = 0.40 * circ_contour + 0.60 * ellipse_circ
            else:
                circ = circ_contour

            # Hough score (optional)
            roi = mask[by:by + bh, bx:bx + bw]
            hough_s = self._hough_score(roi) if CFG.use_hough else 0.0
            if CFG.use_hough:
                circ = 0.55 * circ + 0.45 * hough_s

            hsv_ratio = cv2.countNonZero(roi) / (bw * bh) if bw * bh else 0.0

            # Gradient-weighted score: checks shadow-side shading
            grad_score = self._gradient_score(frame, bx, by, bw, bh)

            conf = (CFG.w_hsv  * min(hsv_ratio / 0.5, 1.0)
                  + CFG.w_circ * circ
                  + CFG.w_grad * grad_score)   # [v12] was hardcoded 0.08

            # Distance from bbox width
            dist = (CFG.ball_diameter_cm * CFG.focal_length_px) / bw if bw else 0.0

            dets.append(Detection(
                x1=bx, y1=by, x2=bx + bw, y2=by + bh,
                cx=bx + bw / 2, cy=by + bh / 2,
                width=bw, height=bh, area=int(area),
                dist_cm=dist, confidence=conf,
                circularity=circ, source="hsv",
                hsv_ratio=hsv_ratio,
                ellipse_a=el_a, ellipse_b=el_b,
            ))
        # Keep only the top candidates to prevent flooding the fuser
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets[:CFG.max_hsv_dets]

    # ── Gradient score — detects diffuse shading of a sphere ─────────────────
    @staticmethod
    def _gradient_score(frame: np.ndarray,
                        bx: int, by: int, bw: int, bh: int) -> float:
        """
        A sphere has a smooth gradient from bright centre to dark edge.
        Compute Laplacian variance inside the ROI: low = smooth shading.
        Returns a score in [0, 1] where 1 = sphere-like smoothness.
        """
        pad = 4
        x1 = max(0, bx - pad); y1 = max(0, by - pad)
        x2 = min(frame.shape[1], bx + bw + pad)
        y2 = min(frame.shape[0], by + bh + pad)
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if roi.size == 0: return 0.0
        lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        # Low Laplacian variance = smooth = sphere-like; cap at 500
        return max(0.0, 1.0 - lap_var / 500.0)

    # ── Enrich YOLO det with HSV + ellipse ───────────────────────────────────
    def _enrich(self, d: Detection, frame: np.ndarray,
                mask: np.ndarray) -> Optional[Detection]:
        """
        Returns None if YOLO box has insufficient HSV coverage —
        this kills face/person/wall detections from YOLO that happen
        to be classified as 'sports ball'.
        """
        roi = mask[d.y1:d.y2, d.x1:d.x2]
        if roi.size == 0: return None

        hsv_ratio = cv2.countNonZero(roi) / d.area if d.area else 0.0

        # Hard gate: if the YOLO box doesn't overlap the tennis-ball
        # HSV mask, it's not a ball — discard regardless of YOLO score.
        if hsv_ratio < CFG.yolo_min_hsv_ratio:
            return None

        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        circ = 0.0; el_a = el_b = 0.0
        for c in cnts:
            cc = self._contour_circularity(c)
            if cc > circ:
                circ = cc
                if len(c) >= CFG.ellipse_fit_min:
                    try:
                        _, (ea, eb), _ = cv2.fitEllipse(c)
                        el_a = max(ea, eb) / 2
                        el_b = min(ea, eb) / 2
                        if el_a > 0:
                            el_circ = el_b / el_a
                            circ = 0.40 * circ + 0.60 * el_circ
                    except Exception:
                        pass

        grad_score = self._gradient_score(frame, d.x1, d.y1, d.width, d.height)

        conf = (CFG.w_yolo * d.confidence
              + CFG.w_hsv  * min(hsv_ratio / 0.5, 1.0)
              + CFG.w_circ * circ
              + CFG.w_grad * grad_score)   # [v12] was hardcoded 0.08

        return dataclasses.replace(
            d, hsv_ratio=hsv_ratio, circularity=circ,
            confidence=min(conf, 1.0),
            ellipse_a=el_a, ellipse_b=el_b,
        )

    # ── Fusion NMS ────────────────────────────────────────────────────────────
    @staticmethod
    def _fuse(yolo: List[Detection],
              hsv:  List[Detection]) -> List[Detection]:
        all_dets = sorted(yolo + hsv, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in all_dets:
            merged = False
            for i, k in enumerate(kept):
                if det.iou(k) > CFG.iou_merge_thresh:
                    if det.source != k.source:
                        # Cross-source agreement: boost confidence
                        boosted = min(k.confidence + 0.10, 1.0)
                        # Prefer ellipse data if available
                        el_a = det.ellipse_a if det.ellipse_a > k.ellipse_a else k.ellipse_a
                        el_b = det.ellipse_b if det.ellipse_b > k.ellipse_b else k.ellipse_b
                        kept[i] = dataclasses.replace(
                            k, confidence=boosted,
                            source="hybrid", ellipse_a=el_a, ellipse_b=el_b,
                        )
                    merged = True
                    break
            if not merged:
                kept.append(det)
        return kept

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _contour_circularity(cnt) -> float:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        return (4 * math.pi * area / peri ** 2) if peri > 0 else 0.0

    @staticmethod
    def _hough_score(roi_mask: np.ndarray) -> float:
        h, w = roi_mask.shape
        if h < 10 or w < 10: return 0.0
        circles = cv2.HoughCircles(
            roi_mask, cv2.HOUGH_GRADIENT,
            dp=CFG.hough_dp,
            minDist=max(20, min(h, w) // 2),
            param1=CFG.hough_param1,
            param2=CFG.hough_param2,
            minRadius=max(CFG.hough_min_r, min(h, w) // 6),
            maxRadius=min(CFG.hough_max_r, min(h, w) // 2 + 2),
        )
        return 1.0 if circles is not None else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FSM  — extended with ALIGN, PICKUP, DEPOSIT states
# ══════════════════════════════════════════════════════════════════════════════
class State(Enum):
    SEARCH   = auto()
    ALIGN    = auto()   # pan to centre ball before approaching
    APPROACH = auto()
    LOCK     = auto()
    PICKUP   = auto()   # arm trajectory execution
    LIFT     = auto()   # [v9] raise arm to clearance height before base rotation
    DEPOSIT  = auto()   # move to basket, release
    VERIFY   = auto()
    SUCCESS  = auto()


@dataclass
class FSMContext:
    state:           State = State.SEARCH
    prev_state:      State = State.SEARCH
    entered_at:      float = field(default_factory=time.monotonic)
    lost_frames:     int   = 0
    lock_frames:     int   = 0
    cx:              float = 0.0
    cy:              float = 0.0
    dist:            float = 0.0
    dist_raw:        float = 0.0   # unfiltered for FSM decisions
    dist_ema:        float = 0.0   # [v6] EMA-smoothed dist for FSM state decisions
    collected:       int   = 0
    n_balls_visible: int   = 0
    # [v6] Detection confirmation: new target must appear N frames before acting
    det_confirm_count: int = 0
    # Arm trajectory state
    arm_traj:        list  = field(default_factory=list)
    arm_step:        int   = 0
    arm_step_at:     float = 0.0
    # [v11] Separate release timer for DEPOSIT so the wait is never measured
    #       against the arm_step_at that was just reset at the top of the step gate.
    deposit_release_at: float = 0.0
    # [v12] Separate grip timer for PICKUP — same fix; arm_step_at is reset by
    #       the outer pickup_step_s gate every 0.8 s, making the 0.5 s grip-hold
    #       check (elif now - arm_step_at >= grip_hold_s) permanently false.
    pickup_grip_at:  float = 0.0
    gripper_closed:  bool  = False
    pickup_ball_x:   float = 0.0   # forward distance at pickup (rover frame, cm)
    pickup_ball_y:   float = 0.0   # lateral offset  at pickup (rover frame, cm)
                                    # y = (err_x_px × dist_ema) / focal_length_px


class TrackingFSM:
    _ENTER: Dict = {}
    _EXIT:  Dict = {}

    def __init__(self, ctx: FSMContext):
        self.ctx = ctx
        self._lk = threading.Lock()

    @classmethod
    def on_enter(cls, s):
        def d(fn): cls._ENTER[s] = fn; return fn
        return d

    @classmethod
    def on_exit(cls, s):
        def d(fn): cls._EXIT[s] = fn; return fn
        return d

    def _transition(self, ns: State):
        ctx = self.ctx
        if ctx.state == ns: return
        if fn := self._EXIT.get(ctx.state): fn(self, ctx)
        log.info("FSM  %s → %s", ctx.state.name, ns.name)
        ctx.prev_state = ctx.state; ctx.state = ns
        ctx.entered_at = time.monotonic()
        ctx.lost_frames = 0; ctx.lock_frames = 0
        if fn := self._ENTER.get(ns): fn(self, ctx)

    def update(self, dets: List[Detection], ack: bool,
               cam_w: int) -> State:
        with self._lk:
            ctx    = self.ctx
            now    = time.monotonic()
            elapsed = now - ctx.entered_at
            target  = dets[0] if dets else None
            ctx.n_balls_visible = len(dets)
            cx_frame = cam_w // 2

            # [v11] Log Arduino ACK/DONE when received during arm states
            if ack and ctx.state in (State.PICKUP, State.LIFT, State.DEPOSIT):
                log.debug("ACK/DONE received in %s (arm_step=%d)", ctx.state.name, ctx.arm_step)

            if ctx.state == State.SUCCESS:
                return ctx.state

            # ── SEARCH ──────────────────────────────────────────────────────
            if ctx.state == State.SEARCH:
                if elapsed > CFG.search_timeout_s and not target:
                    ctx.entered_at = now  # reset timeout, keep scanning
                # [v6] require N consecutive frames before acting on a detection
                confirmed = self._update_confirm(target)
                if confirmed:
                    self._sync(target)
                    self._transition(State.ALIGN)

            # ── ALIGN — pan until ball is centred before driving ────────────
            elif ctx.state == State.ALIGN:
                if target:
                    self._sync(target); ctx.lost_frames = 0
                    err_x = abs(ctx.cx - cx_frame)
                    if err_x <= CFG.align_px_thresh:
                        self._transition(State.APPROACH)
                else:
                    ctx.lost_frames += 1
                    if ctx.lost_frames > CFG.lost_frames_limit:
                        self._transition(State.SEARCH)

            # ── APPROACH — P-controller on distance ────────────────────────
            elif ctx.state == State.APPROACH:
                if target:
                    self._sync(target); ctx.lost_frames = 0
                    # Re-centre if ball drifted significantly
                    if abs(ctx.cx - cx_frame) > CFG.align_px_thresh * 2:
                        self._transition(State.ALIGN)
                    # [v6] use EMA distance for stable lock threshold
                    elif ctx.dist_ema > 0 and ctx.dist_ema <= CFG.dist_lock_cm:
                        self._transition(State.LOCK)
                else:
                    ctx.lost_frames += 1
                    if ctx.lost_frames > CFG.lost_frames_limit or elapsed > CFG.search_timeout_s:
                        self._transition(State.SEARCH)

            # ── LOCK — hold for N frames ────────────────────────────────────
            elif ctx.state == State.LOCK:
                if target:
                    self._sync(target); ctx.lost_frames = 0
                    # [v6] use EMA distance so a single noisy far reading won't abort
                    if ctx.dist_ema > CFG.dist_approach_cm:
                        self._transition(State.APPROACH)
                    else:
                        ctx.lock_frames += 1
                        if ctx.lock_frames >= CFG.lock_hold_frames:
                            # Save ball position for 4-DOF IK
                            ctx.pickup_ball_x = ctx.dist              # [v12] Kalman-filtered best_dist_cm
                                                                      # (was dist_ema: alpha=0.15 too lagged)
                            # Lateral offset via pinhole: y = err_x_px * dist / f
                            err_x_px = ctx.cx - CFG.cam_w / 2
                            ctx.pickup_ball_y = (err_x_px * ctx.dist_ema) / CFG.focal_length_px
                            self._transition(State.PICKUP)
                else:
                    ctx.lost_frames += 1
                    if ctx.lost_frames > CFG.lost_frames_limit or elapsed > CFG.search_timeout_s:
                        self._transition(State.SEARCH)

            # ── PICKUP — execute arm trajectory step by step ────────────────
            elif ctx.state == State.PICKUP:
                # [v9] Retry gate: if enter hook flagged trajectory as bad, go back
                if not ctx.arm_traj:
                    log.info("PICKUP: trajectory aborted by IK retry gate → APPROACH")
                    self._transition(State.APPROACH)
                elif elapsed > CFG.collect_timeout_s:   # [v11] jam / disconnect guard
                    log.warning("PICKUP: collect_timeout (%.1fs) exceeded → APPROACH", elapsed)
                    ctx.gripper_closed = False           # [v12] don't drive with closed claw
                    self._transition(State.APPROACH)
                elif now - ctx.arm_step_at >= CFG.pickup_step_s:
                    ctx.arm_step_at = now
                    if ctx.arm_step < len(ctx.arm_traj):
                        # Caller (ArtemisLink.send) reads arm_traj[arm_step]
                        ctx.arm_step += 1
                    else:
                        # Trajectory complete — close gripper and hold
                        if not ctx.gripper_closed:
                            ctx.gripper_closed = True
                            ctx.pickup_grip_at = now   # [v12] separate timer (not arm_step_at)
                        elif now - ctx.pickup_grip_at >= CFG.grip_hold_s:
                            # [v12] Grasp verify MOVED TO LIFT: while the arm is still
                            # at ground level YOLO sees the ball inside the open claw
                            # and falsely reports a close-range detection → every real
                            # grasp was flagged as a failure and aborted.
                            self._transition(State.LIFT)

            # ── LIFT — raise arm to clearance height before deposit rotation ──
            elif ctx.state == State.LIFT:
                if not ctx.arm_traj:                        # [v11] IK abort gate
                    log.info("LIFT: trajectory aborted by IK gate → APPROACH")
                    ctx.gripper_closed = False              # [v12] open claw on abort
                    self._transition(State.APPROACH)
                elif elapsed > CFG.collect_timeout_s:       # [v11] hang guard
                    log.warning("LIFT: collect_timeout (%.1fs) exceeded → APPROACH", elapsed)
                    ctx.gripper_closed = False              # [v12] open claw on timeout
                    self._transition(State.APPROACH)
                elif now - ctx.arm_step_at >= CFG.pickup_step_s:
                    ctx.arm_step_at = now
                    if ctx.arm_step < len(ctx.arm_traj):
                        ctx.arm_step += 1
                    else:
                        # [v12] Grasp verify moved HERE from PICKUP: arm is raised so
                        # YOLO's view of the ball on the ground is obscured by the
                        # claw.  A ball still visible close-up means it fell out.
                        if (target is not None and
                                target.best_dist_cm <= CFG.dist_lock_cm * 1.5):
                            log.warning(
                                "LIFT: ball still visible at %.1f cm after raising "
                                "— grasp failed, retrying APPROACH.",
                                target.best_dist_cm)
                            ctx.gripper_closed = False
                            self._transition(State.APPROACH)
                        else:
                            self._transition(State.DEPOSIT)

            # ── DEPOSIT — move to box, release ─────────────────────────────
            elif ctx.state == State.DEPOSIT:
                if not ctx.arm_traj:                        # [v11] IK abort gate
                    log.info("DEPOSIT: trajectory aborted by IK gate → APPROACH")
                    ctx.gripper_closed = False              # [v12] open claw on abort
                    self._transition(State.APPROACH)
                elif elapsed > CFG.collect_timeout_s:       # [v11] hang guard
                    log.warning("DEPOSIT: collect_timeout (%.1fs) exceeded → APPROACH", elapsed)
                    ctx.gripper_closed = False              # [v12] open claw on timeout
                    self._transition(State.APPROACH)
                elif now - ctx.arm_step_at >= CFG.deposit_step_s:
                    ctx.arm_step_at = now
                    if ctx.arm_step < len(ctx.arm_traj):
                        ctx.arm_step += 1
                    else:
                        if ctx.gripper_closed:
                            ctx.gripper_closed = False
                            # [v11] Use a SEPARATE timer so the release-wait is
                            # not measured against arm_step_at which was just
                            # reset at the top of this block (the bug: 0 < 0.4
                            # was always False, so collected never incremented).
                            ctx.deposit_release_at = now
                        elif now - ctx.deposit_release_at >= CFG.grip_hold_s:
                            # [v11] was magic literal 0.4 — now uses CFG.grip_hold_s
                            ctx.collected += 1
                            self._transition(State.VERIFY)

            # ── VERIFY — re-scan after collection ──────────────────────────
            elif ctx.state == State.VERIFY:
                if elapsed >= CFG.verify_duration_s:
                    # [T3] Only SUCCESS when we have reached the target count
                    if ctx.collected >= CFG.target_count:
                        self._transition(State.SUCCESS)
                    elif target:
                        self._sync(target)
                        self._transition(State.APPROACH)
                    else:
                        self._transition(State.SEARCH)

            return ctx.state

    def _sync(self, d: Detection):
        self.ctx.cx      = d.cx
        self.ctx.cy      = d.cy
        self.ctx.dist    = d.best_dist_cm      # improved formula
        self.ctx.dist_raw = d.dist_cm
        # [v6] EMA-smooth distance for FSM state-transition decisions
        alpha = CFG.fsm_dist_ema_alpha
        if self.ctx.dist_ema == 0.0:
            self.ctx.dist_ema = d.best_dist_cm   # cold-start
        else:
            self.ctx.dist_ema = alpha * d.best_dist_cm + (1.0 - alpha) * self.ctx.dist_ema

    def _update_confirm(self, target: Optional["Detection"]) -> bool:
        """
        [v6] Returns True only if `target` has been seen in
        CFG.det_confirm_frames consecutive frames.
        Resets the counter when a target disappears.
        """
        ctx = self.ctx
        if target is None:
            ctx.det_confirm_count = 0
            return False
        ctx.det_confirm_count = min(ctx.det_confirm_count + 1, CFG.det_confirm_frames)
        return ctx.det_confirm_count >= CFG.det_confirm_frames


# ── FSM enter/exit hooks ─────────────────────────────────────────────────────
@TrackingFSM.on_enter(State.SEARCH)
def _(f, c): log.info("SEARCH — scanning for ball.")

@TrackingFSM.on_enter(State.ALIGN)
def _(f, c): log.info("ALIGN — centering ball, err=%+.0fpx from centre.", c.cx - CFG.cam_w / 2)

@TrackingFSM.on_enter(State.APPROACH)
def _(f, c): log.info("APPROACH — driving, dist=%.1fcm.", c.dist)

@TrackingFSM.on_enter(State.LOCK)
def _(f, c): log.info("LOCK — holding at %.1fcm.", c.dist)

@TrackingFSM.on_enter(State.PICKUP)
def _(f, c):
    log.info("PICKUP — ball at ~%.1fcm fwd, %.1fcm lat. Building trajectory.",
             c.pickup_ball_x, c.pickup_ball_y)
    traj = build_pickup_trajectory(ball_x_cm=c.pickup_ball_x, ball_y_cm=c.pickup_ball_y)
    # [v9] IK retry gate; [v12] consistent set-in-else pattern (was set-then-clear)
    none_count = sum(1 for wp in traj if wp is None)
    if traj and none_count / len(traj) > 0.30:
        log.warning("PICKUP: %d/%d waypoints unreachable (>30%%). "
                    "Clearing trajectory → will fall back to APPROACH.",
                    none_count, len(traj))
        c.arm_traj = []
    else:
        c.arm_traj = traj
        log.info("  PICKUP trajectory: %d waypoints (%d unreachable).", len(traj), none_count)
    c.arm_step       = 0
    c.arm_step_at    = time.monotonic()
    c.pickup_grip_at = 0.0   # [v12] reset separate grip timer on each PICKUP entry
    c.gripper_closed = False

@TrackingFSM.on_enter(State.LIFT)
def _(f, c):
    log.info("LIFT — raising arm to clearance height before deposit rotation.")
    traj = build_lift_trajectory(ball_x_cm=c.pickup_ball_x, ball_y_cm=c.pickup_ball_y)
    # [v11] IK abort gate — mirrors PICKUP gate; prevents silent 0,0,0,0 replay
    none_count = sum(1 for wp in traj if wp is None)
    if traj and none_count / len(traj) > 0.30:
        log.warning("LIFT: %d/%d waypoints unreachable (>30%%). "
                    "Clearing trajectory → will fall back to APPROACH.",
                    none_count, len(traj))
        c.arm_traj = []
    else:
        c.arm_traj = traj
        log.info("  LIFT trajectory: %d waypoints (%d unreachable).", len(traj), none_count)
    c.arm_step   = 0
    c.arm_step_at = time.monotonic()

@TrackingFSM.on_enter(State.DEPOSIT)
def _(f, c):
    log.info("DEPOSIT — moving to basket. Collected so far: %d.", c.collected)
    traj = build_deposit_trajectory(ball_x_cm=c.pickup_ball_x, ball_y_cm=c.pickup_ball_y)
    # [v11] IK abort gate — mirrors PICKUP gate; prevents silent 0,0,0,0 replay
    none_count = sum(1 for wp in traj if wp is None)
    if traj and none_count / len(traj) > 0.30:
        log.warning("DEPOSIT: %d/%d waypoints unreachable (>30%%). "
                    "Clearing trajectory → will fall back to APPROACH.",
                    none_count, len(traj))
        c.arm_traj = []
    else:
        c.arm_traj = traj
        log.info("  DEPOSIT trajectory: %d waypoints (%d unreachable).", len(traj), none_count)
    c.arm_step          = 0
    c.arm_step_at       = time.monotonic()
    c.deposit_release_at = 0.0   # [v11] reset release timer on every DEPOSIT entry

@TrackingFSM.on_enter(State.VERIFY)
def _(f, c): log.info("VERIFY — re-scanning. Collected: %d.", c.collected)

@TrackingFSM.on_enter(State.SUCCESS)
def _(f, c): log.info("★★★ SUCCESS — %d/%d ball(s) collected!", c.collected, CFG.target_count)


# ══════════════════════════════════════════════════════════════════════════════
# Serial / ArtemisLink
# ══════════════════════════════════════════════════════════════════════════════

class ArtemisLink:
    """
    Sends command packets to Arduino/microcontroller over serial.

    Packet format (CSV + newline) — [v11 blueprint-exact]:
        LeftSpeed, RightSpeed, Q1, Q2, Q3, Q4, Gripper

        LeftSpeed / RightSpeed  — wheel PWM computed on Pi [0-255 forward,
                                  negative values indicate reverse if motor
                                  driver supports it; clamp to [-255,255]]
        Q1..Q4                  — joint angles in degrees (0.0 when idle)
        Gripper                 — 1 = closed, 0 = open

    Arduino executes the command directly with no additional logic.

    Receives:
        "ACK\\n"  — arm step acknowledged
        "DONE\\n" — collection confirmed
    """

    def __init__(self):
        self._log = logging.getLogger("ArtemisLink")
        self._ser = None
        self._q   = queue.Queue(maxsize=CFG.serial_queue_size)
        self._ack = threading.Event()
        self._run = True
        # [T3] PD steering state — [v11] _prev_err_x is None until first send()
        # so the derivative term is skipped on the very first frame (avoids a
        # −192 PWM spike when ctx.cx=0 but the real ball is at x=320).
        self._prev_err_x:  Optional[float] = None
        self._last_send_t: float = time.monotonic()

        if CFG.no_serial or not _SERIAL_OK:
            self._log.info("Vision-only mode (no serial).")
            return
        threading.Thread(target=self._tx, daemon=True, name="SerTX").start()
        threading.Thread(target=self._rx, daemon=True, name="SerRX").start()

    @staticmethod
    def list_ports() -> List[str]:
        if not _SERIAL_OK: return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def send(self, ctx: FSMContext):
        if CFG.no_serial or not _SERIAL_OK: return

        err_x = ctx.cx - CFG.cam_w / 2   # pixels; +ve = ball right of centre

        # ── [T3] PD wheel speed controller ──────────────────────────────────
        # Proportional + Derivative on pixel error.
        # Arduino receives ready-to-use PWM values and applies them directly.
        now_s  = time.monotonic()
        dt_s   = max(now_s - self._last_send_t, 1e-3)
        self._last_send_t = now_s
        # [v11] Skip derivative on first call (prev is None) to prevent the
        # −192 PWM spike from ctx.cx=0 vs frame_centre=320 on cold start.
        if self._prev_err_x is None:
            derr_x = 0.0
        else:
            derr_x = (err_x - self._prev_err_x) / dt_s
        self._prev_err_x = err_x

        if ctx.state in (State.APPROACH, State.LOCK):
            steer     = CFG.drive_kp * err_x + CFG.drive_kd * derr_x
            left_pwm  = int(_clamp(CFG.drive_base_speed + steer, 0, 255))
            right_pwm = int(_clamp(CFG.drive_base_speed - steer, 0, 255))
        elif ctx.state in (State.SEARCH, State.ALIGN):
            steer     = CFG.drive_kp * err_x + CFG.drive_kd * derr_x
            left_pwm  = int(_clamp( steer, -255, 255))
            right_pwm = int(_clamp(-steer, -255, 255))
        else:
            left_pwm = right_pwm = 0
            self._prev_err_x = None   # [v11] reset to None: derivative skipped on next movement start

        # ── [v9] Extract all 4 joint angles ──────────────────────────────────
        q1 = q2 = q3 = q4 = 0.0
        gripper = 1 if ctx.gripper_closed else 0
        if ctx.state in (State.PICKUP, State.LIFT, State.DEPOSIT) and ctx.arm_traj:
            idx    = min(ctx.arm_step, len(ctx.arm_traj) - 1)
            angles = ctx.arm_traj[idx]
            if angles is not None:
                q1, q2, q3, q4 = angles   # 4-tuple from ik_4dof

        cmd = (
            f"{left_pwm},{right_pwm},"
            f"{q1:.1f},{q2:.1f},{q3:.1f},{q4:.1f},{gripper}\n"
        ).encode()

        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            try: self._q.get_nowait(); self._q.put_nowait(cmd)
            except queue.Empty: pass

    def consume_ack(self) -> bool:
        got = self._ack.is_set()
        if got: self._ack.clear()
        return got

    def stop(self):
        self._run = False
        if self._ser:
            try: self._ser.close()
            except: pass

    @property
    def connected(self) -> bool:
        return (not CFG.no_serial and _SERIAL_OK and
                self._ser is not None and self._ser.is_open)

    def _connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                CFG.serial_port, CFG.serial_baud, timeout=CFG.serial_timeout
            )
            self._log.info("Serial connected: %s @ %d baud",
                           CFG.serial_port, CFG.serial_baud)
            return True
        except Exception as e:
            self._log.warning("Serial connect failed: %s", e)
            ports = self.list_ports()
            if ports:
                self._log.info("Available ports: %s  (--port <name>)", ports)
            return False

    def _tx(self):
        backoff = CFG.serial_reconnect_base
        while self._run:
            if not self.connected:
                if self._connect(): backoff = CFG.serial_reconnect_base
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, CFG.serial_reconnect_max)
                    continue
            try:
                payload = self._q.get(timeout=0.5)
                if self._ser and self._ser.is_open:
                    self._ser.write(payload)
            except queue.Empty:
                pass
            except Exception as e:
                self._log.error("TX error: %s", e)
                try: self._ser and self._ser.close()
                except: pass
                self._ser = None

    def _rx(self):
        while self._run:
            if not self.connected: time.sleep(0.1); continue
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line in ("ACK", "DONE"):
                    self._ack.set()
            except: time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Camera  — threaded, always-latest frame
# ══════════════════════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self):
        if sys.platform == "win32":
            # Use DirectShow on Windows to prevent MSMF crashes
            self._cap = cv2.VideoCapture(CFG.cam_src, cv2.CAP_DSHOW)
        else:
            # Use standard capture on Raspberry Pi / Linux
            self._cap = cv2.VideoCapture(CFG.cam_src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # [v7-fix] Log actual vs requested resolution immediately after open.
        # On Raspberry Pi, libcamera/v4l2 may silently ignore CAP_PROP_FRAME_WIDTH
        # requests and return native sensor resolution instead, which corrupts all
        # pixel-width-to-distance calculations (focal_length_px was calibrated at
        # 640×480; a different actual resolution produces wrong dist_cm values and
        # hard-clips the working range, e.g. 200 cm reported as only ~40 cm).
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera: source=%d  requested=%dx%d  actual=%dx%d",
                 CFG.cam_src, CFG.cam_w, CFG.cam_h, actual_w, actual_h)
        if actual_w != CFG.cam_w or actual_h != CFG.cam_h:
            log.warning(
                "RESOLUTION MISMATCH: camera delivered %dx%d, not the requested %dx%d. "
                "focal_length_px=%.1f was calibrated at %dx%d — distance estimates will "
                "be wrong at the current resolution. Options: (a) re-run --calibrate "
                "at %dx%d, or (b) fix cam_w/cam_h in Config to match actual resolution.",
                actual_w, actual_h, CFG.cam_w, CFG.cam_h,
                CFG.focal_length_px, CFG.cam_w, CFG.cam_h,
                actual_w, actual_h,
            )

        # Warm-up: read frames until auto-exposure settles OR 3 s timeout.
        # The Brio 100 (and most USB cameras) initialise at very low gain and
        # take up to ~60 frames (~2 s) before AE converges to a stable value.
        # We also fall back to a frame-count cap so startup is never infinite.
        # [v12] Removed local `import time/cv2/numpy` — module-level imports used.
        _t0 = time.monotonic()
        for _ in range(90):          # cap at 90 frames (~3 s at 30 fps)
            ok, _f = self._cap.read()
            if ok and _f is not None:
                mean_v = cv2.cvtColor(_f, cv2.COLOR_BGR2GRAY).mean()
                if mean_v >= 40.0:   # AE has settled to a usable brightness
                    break
            if time.monotonic() - _t0 > 3.0:
                break

        self._ret, self._frame = self._cap.read()
        self._lock = threading.Lock()
        self._run  = True
        threading.Thread(target=self._loop, daemon=True, name="Cam").start()

    def _loop(self):
        while self._run:
            r, f = self._cap.read()
            with self._lock:
                self._ret, self._frame = r, f

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._ret, (self._frame.copy() if self._ret else None)

    def downscale(self, w: int, h: int):
        """Reduce resolution at runtime if FPS drops.
        [v11] Also syncs CFG.cam_w/cam_h so err_x frame-centre and
        align_px_thresh are computed against the new resolution, not the
        calibrated 640 px width — avoids a constant ~80 px steer bias.
        """
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        CFG.cam_w = w   # [v11] keep Config in sync with actual camera resolution
        CFG.cam_h = h
        log.warning("Camera downscaled to %dx%d for FPS recovery. CFG updated.", w, h)

    def release(self):
        self._run = False
        time.sleep(0.1)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════
class HUD:
    _F  = cv2.FONT_HERSHEY_SIMPLEX
    _PC = {
        State.SEARCH:  (80,  80,  80),
        State.ALIGN:   (200, 180,  0),
        State.APPROACH:(0,  165, 255),
        State.LOCK:    (0,  220,   0),
        State.PICKUP:  (0,  200, 220),
        State.LIFT:    (120, 220, 255),   # [v9] light blue — lifting
        State.DEPOSIT: (200, 120, 255),
        State.VERIFY:  (200, 200,   0),
        State.SUCCESS: (0,  255, 128),
    }
    _SC = {"yolo":  (255, 180, 0),
           "hsv":   (0,   200, 255),
           "hybrid":(0,   255, 120)}

    @classmethod
    def draw(cls, frame: np.ndarray, ctx: FSMContext,
             dets: List[Detection], fps: float, serial_ok: bool):
        h, w = frame.shape[:2]
        sc   = cls._PC[ctx.state]
        cx_f, cy_f = w // 2, h // 2

        for i, d in enumerate(dets):
            is_t  = (i == 0)
            col   = sc if is_t else (100, 100, 100)
            thick = 2  if is_t else 1
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), col, thick)
            # Source dot
            cv2.circle(frame, (d.x1 + 6, d.y1 + 6), 4,
                       cls._SC.get(d.source, (255, 255, 255)), -1)
            # Ellipse overlay if fitted
            if d.ellipse_a > 0 and d.ellipse_b > 0:
                ex = int(d.cx); ey = int(d.cy)
                cv2.ellipse(frame, (ex, ey),
                            (int(d.ellipse_a), int(d.ellipse_b)),
                            0, 0, 360, (0, 200, 100), 1)
            lbl = (f"#{i+1} {d.best_dist_cm:.0f}cm "
                   f"c={d.confidence:.2f} "
                   f"{'[T]' if is_t else ''}")
            cv2.putText(frame, lbl, (d.x1, d.y1 - 6), cls._F, 0.38, col, 1)
            # Circularity bar
            bw = int(d.circularity * (d.x2 - d.x1))
            cv2.rectangle(frame,
                          (d.x1, d.y2 + 2), (d.x1 + bw, d.y2 + 6),
                          (80, 200, 80), -1)
            if is_t:
                cv2.drawMarker(frame, (int(d.cx), int(d.cy)),
                               sc, cv2.MARKER_CROSS, 20, 2)
                cv2.arrowedLine(frame, (cx_f, cy_f),
                                (int(d.cx), int(d.cy)),
                                (200, 200, 0), 1, tipLength=0.12)

        # LOCK progress bar
        if ctx.state == State.LOCK:
            p = min(ctx.lock_frames / CFG.lock_hold_frames, 1.0)
            cv2.rectangle(frame, (20, h - 18), (w - 20, h - 8), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, h - 18),
                          (20 + int((w - 40) * p), h - 8), (0, 220, 0), -1)
            cv2.putText(frame, "LOCK HOLD", (20, h - 22),
                        cls._F, 0.38, (0, 220, 0), 1)

        # PICKUP / LIFT / DEPOSIT arm progress bar
        if ctx.state in (State.PICKUP, State.LIFT, State.DEPOSIT) and ctx.arm_traj:
            p = min(ctx.arm_step / max(len(ctx.arm_traj), 1), 1.0)
            bar_col = cls._PC[ctx.state]
            cv2.rectangle(frame, (20, h - 18), (w - 20, h - 8), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, h - 18),
                          (20 + int((w - 40) * p), h - 8), bar_col, -1)
            label = f"ARM {ctx.state.name} step {ctx.arm_step}/{len(ctx.arm_traj)}"
            cv2.putText(frame, label, (20, h - 22), cls._F, 0.38, bar_col, 1)

        # State banner
        banner = ctx.state.name
        if ctx.state == State.SUCCESS:
            banner = f"SUCCESS — {ctx.collected}/{CFG.target_count} balls collected!"
        cv2.rectangle(frame, (0, 0), (380, 30), (15, 15, 15), -1)
        cv2.putText(frame, banner, (8, 22), cls._F, 0.72, sc, 2)

        # Telemetry (top-right)
        rows = [
            (f"FPS  {fps:5.1f}",                (190, 190, 190)),
            (f"SER  {'OK' if serial_ok else '--'}", (0, 200, 0) if serial_ok else (120, 120, 120)),
            (f"BALL {ctx.n_balls_visible}",        (200, 200, 0)),
            (f"COLL {ctx.collected}/{CFG.target_count}", (0, 220, 120)),
            (f"DIST {ctx.dist:5.1f}cm",            (200, 200, 200)),
            (f"ERR  {int(ctx.cx - cx_f):+d}px",   (180, 180, 255)),
            (f"GRIP {'CLOSED' if ctx.gripper_closed else 'open'}",
             (0, 255, 200) if ctx.gripper_closed else (120, 120, 120)),
        ]
        for i, (t, c) in enumerate(rows):
            cv2.putText(frame, t, (w - 165, 20 + i * 18), cls._F, 0.43, c, 1)

        # Crosshair
        cv2.line(frame, (cx_f - 14, cy_f), (cx_f + 14, cy_f), (60, 60, 60), 1)
        cv2.line(frame, (cx_f, cy_f - 14), (cx_f, cy_f + 14), (60, 60, 60), 1)


# ══════════════════════════════════════════════════════════════════════════════
# FPS Watchdog  — auto-reduce resolution on Pi if FPS crashes
# ══════════════════════════════════════════════════════════════════════════════
class FPSWatchdog:
    _FALLBACK_SIZES = [(480, 360), (320, 240)]

    def __init__(self, cam: "CameraStream"):
        self._cam      = cam
        self._fps_buf: List[float] = []
        self._low_since: Optional[float] = None
        self._idx = 0

    def update(self, fps: float):
        self._fps_buf.append(fps)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)
        if len(self._fps_buf) < 10: return
        avg = sum(self._fps_buf) / len(self._fps_buf)
        if avg < CFG.fps_target:
            if self._low_since is None:
                self._low_since = time.monotonic()
            elif time.monotonic() - self._low_since > CFG.fps_low_s:
                if sys.platform != "win32":  # Only downscale on the Pi!
                    self._downscale()
        else:
            self._low_since = None

    def _downscale(self):
        if self._idx >= len(self._FALLBACK_SIZES): return
        w, h = self._FALLBACK_SIZES[self._idx]
        self._cam.downscale(w, h)
        self._idx += 1
        self._low_since = None
        self._fps_buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Main tracker
# ══════════════════════════════════════════════════════════════════════════════
class AutoPanTracker:
    def __init__(self):
        self._done = threading.Event()
        signal.signal(signal.SIGINT,  lambda *_: self._done.set())
        signal.signal(signal.SIGTERM, lambda *_: self._done.set())

        # ── Select inference engine ────────────────────────────────────────────
        import os
        engine = None

        # 1. Prefer TFLite if available
        if _TFLITE_OK and os.path.exists(CFG.tflite_path):
            log.info("Using TFLite INT8 model (Fastest!).")
            engine = TFLiteEngine(CFG.tflite_path)

        # 2. Fallback to ONNX
        elif _ONNX_OK:
            if CFG.use_int8 and os.path.exists(CFG.onnx_int8_path):
                log.info("Using INT8 ONNX model.")
                engine = ONNXEngine(CFG.onnx_int8_path)
            elif os.path.exists(CFG.onnx_path):
                log.info("Using standard ONNX model.")
                engine = ONNXEngine(CFG.onnx_path)

        # 3. Fallback to PyTorch
        if engine is None and _YOLO_OK:
            log.warning("TFLite/ONNX not found — falling back to slow PyTorch.")

            _pt = _YOLO(CFG.pt_path)

            class _PTEngine:
                def predict(self_, f):
                    res = _pt.predict(f, classes=[CFG.yolo_class],
                                       imgsz=CFG.infer_imgsz,
                                       conf=CFG.infer_conf,
                                       iou=CFG.infer_iou,
                                       half=False,
                                       agnostic_nms=True,
                                       verbose=False)
                    dets = []
                    for r in res:
                        for b in r.boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            w2 = x2 - x1; h2 = y2 - y1; area = w2 * h2
                            if area < CFG.min_box_area: continue
                            if area > CFG.max_box_area: continue   # [v12] mirrors ONNX/TFLite fix
                            dist = (CFG.ball_diameter_cm * CFG.focal_length_px) / w2 if w2 else 0
                            dets.append(Detection(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                                width=w2, height=h2, area=area,
                                dist_cm=dist,
                                confidence=float(b.conf[0]),
                                circularity=0.0, source="yolo",
                                hsv_ratio=0.0,
                            ))
                    return dets
            dummy = np.zeros((CFG.infer_imgsz, CFG.infer_imgsz, 3), np.uint8)
            _pt.predict(dummy, imgsz=CFG.infer_imgsz, verbose=False)
            engine = _PTEngine()
        if engine is None:
            raise RuntimeError("Neither onnxruntime nor ultralytics installed.")

        self._worker   = AsyncInferenceWorker(engine)
        self._detector = BallDetector(self._worker)
        self._ctx      = FSMContext()
        self._ctx.cx   = float(CFG.cam_w // 2)   # [v12] prevent cold-start spin: err_x=0 at startup
        self._fsm      = TrackingFSM(self._ctx)
        self._comm     = ArtemisLink()

        # Kalman filters for cx, cy, dist
        self._kx = KalmanFilter1D(CFG.kalman_proc, CFG.kalman_meas_xy)
        self._ky = KalmanFilter1D(CFG.kalman_proc, CFG.kalman_meas_xy)
        self._kd = KalmanFilter1D(CFG.kalman_proc, CFG.kalman_meas_dist)
        self._prev_cx: Optional[float] = None
        self._fps = 30.0
        self._t0  = time.monotonic()

    def _reset_kalman_if_jumped(self, det: Optional[Detection]):
        if det is None:
            self._kx.reset(); self._ky.reset(); self._kd.reset()
            self._prev_cx = None; return
        # [v6] lowered jump threshold: at 2 m the ball is small, 90 px jumps are
        # less likely — 60 px is a more appropriate false-detection guard.
        # [v12] raised to 120 px: at ≤15 cm the ball fills most of the frame and
        # legitimate frame-to-frame movement easily exceeds 60 px, causing the
        # Kalman to reset every frame and thrash during the critical close-approach.
        if self._prev_cx is not None and abs(det.cx - self._prev_cx) > 120:
            self._kx.reset(); self._ky.reset(); self._kd.reset()
        self._prev_cx = det.cx

    def run(self):
        cam      = CameraStream()
        watchdog = FPSWatchdog(cam)
        log.info("AutoPan v12 running. Press Q to quit.")

        try:
            while not self._done.is_set():
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.005); continue

                now   = time.monotonic()
                dt    = max(now - self._t0, 1e-6); self._t0 = now
                self._fps = 0.9 * self._fps + 0.1 / dt
                watchdog.update(self._fps)

                raw_dets = self._detector.detect(frame)

                # Kalman smooth nearest target
                target = raw_dets[0] if raw_dets else None
                self._reset_kalman_if_jumped(target)
                dets = list(raw_dets)
                if target and dets:
                    # [v12] Feed Kalman the fused best_dist_cm (avg bbox+ellipse),
                    # not raw dist_cm (bbox-only). Then zero ellipse_b so that
                    # best_dist_cm == dist_cm (avoids double-averaging the ellipse).
                    smoothed_dist = self._kd.update(target.best_dist_cm, dt)
                    dets[0] = dataclasses.replace(
                        target,
                        cx        = self._kx.update(target.cx, dt),
                        cy        = self._ky.update(target.cy, dt),
                        dist_cm   = smoothed_dist,
                        ellipse_b = 0.0,
                    )

                ack = self._comm.consume_ack()
                self._fsm.update(dets, ack, cam_w=CFG.cam_w)
                self._comm.send(self._ctx)

                HUD.draw(frame, self._ctx, dets, self._fps, self._comm.connected)
                cv2.imshow("AutoPan v12", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._done.set()

        except Exception:
            log.exception("Fatal error")
        finally:
            cam.release()
            self._worker.stop()
            self._comm.stop()
            cv2.destroyAllWindows()
            log.info("Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Calibration wizard
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration():
    dist = float(input("Known distance to ball (cm): "))
    cap  = cv2.VideoCapture(CFG.cam_src)
    # [v12] Set resolution to match runtime so focal_length_px is valid at that res
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Calibration: camera at %dx%d (requested %dx%d).",
             actual_w, actual_h, CFG.cam_w, CFG.cam_h)
    from ultralytics import YOLO as _Y
    mdl = _Y(CFG.pt_path)
    print("Press SPACE to capture at known distance, Q to abort.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        res = mdl.predict(frame, classes=[CFG.yolo_class], verbose=False)
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                w = x2 - x1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"w={w}px", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Calibration", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            for r in res:
                for b in r.boxes:
                    w = int(b.xyxy[0][2]) - int(b.xyxy[0][0])
                    fl = (w * dist) / CFG.ball_diameter_cm
                    print(f"\n  focal_length_px = {fl:.1f}")
                    print(f"  Set CFG.focal_length_px = {fl:.1f}  in Config above.\n")
            break
        if k == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# IK test utility
# ══════════════════════════════════════════════════════════════════════════════
def run_ik_test():
    """Quick sanity check for the 4-DOF IK solver over a grid of positions."""
    print("\n=== IK Sanity Test (4-DOF) ===")
    # (x_fwd_cm, y_lat_cm, z_cm)
    test_pts = [
        (10, 0.0, ARM.pickup_z_cm),
        (15, 0.0, ARM.pickup_z_cm),
        (20, 0.0, ARM.pickup_z_cm),
        (10, 5.0, ARM.pickup_z_cm),
        (15, 3.0, ARM.pickup_z_cm),
        (ARM.basket_x, ARM.basket_y, ARM.basket_z),  # deposit position
    ]
    for (tx, ty, tz) in test_pts:
        r = ik_4dof(tx, ty, tz)
        if r:
            q1, q2, q3, q4 = r
            print(f"  ({tx:6.1f},{ty:5.1f},{tz:5.1f}) cm → "
                  f"q1={q1:7.1f}° q2={q2:7.1f}° q3={q3:7.1f}° q4={q4:7.1f}°")
        else:
            print(f"  ({tx:6.1f},{ty:5.1f},{tz:5.1f}) cm → UNREACHABLE")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AutoPan v12")
    ap.add_argument("--calibrate",    action="store_true",
                    help="Run focal-length calibration wizard")
    ap.add_argument("--ik-test",      action="store_true",
                    help="Run IK solver sanity test and exit")
    ap.add_argument("--no-serial",    action="store_true")
    ap.add_argument("--port",         default=CFG.serial_port)
    ap.add_argument("--cam",          type=int,   default=CFG.cam_src)
    ap.add_argument("--imgsz",        type=int,   default=CFG.infer_imgsz)
    ap.add_argument("--focal",        type=float, default=CFG.focal_length_px)
    ap.add_argument("--debug-hsv",    action="store_true")
    ap.add_argument("--hough",        action="store_true",
                    help="Enable Hough circles (more accurate, ~8ms slower)")
    ap.add_argument("--int8",         action="store_true",
                    help="Use INT8 quantised ONNX model (fastest on Pi)")
    ap.add_argument("--conf",         type=float, default=CFG.infer_conf,
                    help="YOLO confidence threshold (default 0.30)")
    ap.add_argument("--target-count", type=int,   default=CFG.target_count,
                    help="Number of balls to collect before SUCCESS (default 3)")
    args = ap.parse_args()

    CFG.serial_port     = args.port
    CFG.cam_src         = args.cam
    CFG.infer_imgsz     = args.imgsz
    CFG.focal_length_px = args.focal
    CFG.debug_hsv       = args.debug_hsv
    CFG.use_hough       = args.hough
    CFG.no_serial       = getattr(args, "no_serial", False)
    CFG.use_int8        = args.int8
    CFG.infer_conf      = args.conf
    CFG.target_count    = args.target_count

    if args.ik_test:
        run_ik_test()
    elif args.calibrate:
        run_calibration()
    else:
        AutoPanTracker().run()
