#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Two-block PsychoPy experiment.

Block 1
-------
1. Present one white disc at 8 or 30 deg eccentricity.
2. Press Space to remove the disc.
3. Record Up/Down key presses.
4. Press Space again to end the trial.

Block 2
-------
1. Present one white disc at 8 or 30 deg eccentricity on the presentation screen.
2. Press Space to remove the disc.
3. The presentation screen becomes completely black.
4. On the second screen, present a blurred probe disc.
5. Participant adjusts probe blur with Up/Down.
6. Press Space to confirm blur.
7. Participant adjusts probe luminance / grey level with Up/Down.
8. Press Space to confirm luminance and end the trial.
"""
from __future__ import annotations

import platform
import subprocess
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from psychopy import core, gui, visual, monitors
from psychopy.hardware import keyboard

if platform.system() == "Windows":
    import winsound


# =========================================================
# General experiment settings
# =========================================================

EXP_NAME = "disc_keylog_then_blur_luminance_adjustment"

# Real experiment with two monitors:
FULLSCREEN = True
PILOT_MODE = False

STIM_SCREEN_INDEX = 0 
PROBE_SCREEN_INDEX =  1

STIM_WINDOW_SIZE = [1000, 1000]
PROBE_WINDOW_SIZE = [800, 800]

BACKGROUND_COLOR = [-1, -1, -1]

# --- Global Physical Setup (137-degree inward wrap) ---
VIEWING_DISTANCE_CM = 50.0 

# --- Monitor settings for the stimulus screen (LEFT) ---
MONITOR_WIDTH_CM = 56.0
STIM_BEZEL_WIDTH_CM = 2.5
SCREEN_RESOLUTION_PX = [1600, 900]

# --- Monitor settings for the probe screen (RIGHT) ---
PROBE_MONITOR_WIDTH_CM = 53.0 
PROBE_BEZEL_WIDTH_CM = 0.5    


# =========================================================
# Trial settings
# =========================================================

BLOCK1_N_TRIALS = 6
BLOCK2_N_TRIALS = 6

ECCENTRICITIES_DEG = [8.0, 30.0]
BALANCED_ECCENTRICITY = True

DISC_RADIUS_DEG = 5.0 
DISC_COLOR = [1, 1, 1]

SHOW_FIXATION = False 
FIXATION_SIZE_DEG = 0.35


# =========================================================
# Blur adjustment settings (PURE PIXEL BASED)
# =========================================================

# 1. Dynamically calculate the perfect pixel size based on distance
_theta = np.radians(DISC_RADIUS_DEG)
_radius_cm = VIEWING_DISTANCE_CM * np.tan(_theta)
_px_per_cm = 1920.0 / PROBE_MONITOR_WIDTH_CM
PROBE_RADIUS_PIX = int(_radius_cm * _px_per_cm) 

# 2. Canvas increased to 1024 so the heavy blur never gets cut off at the edges
PROBE_IMAGE_RES_PIX = 1024 
PROBE_SIZE_PIX = 1024      

PROBE_INITIAL_BLUR_SIGMA = 5.0
PROBE_BLUR_STEP = 1.0
PROBE_MIN_BLUR = 0.0
PROBE_MAX_BLUR = 40.0

SHOW_PROBE_INSTRUCTIONS = True
SHOW_BLUR_VALUE_FOR_DEBUG = False


# =========================================================
# Luminance adjustment settings
# =========================================================

PROBE_INITIAL_LUMINANCE = 0.50
PROBE_LUMINANCE_STEP = 0.05
PROBE_MIN_LUMINANCE = 0.00
PROBE_MAX_LUMINANCE = 1.00

SHOW_LUMINANCE_VALUE_FOR_DEBUG = False
LUMINANCE_ADJUSTMENT_BLUR_SIGMA = 0.0


# =========================================================
# Sound settings
# =========================================================

PLAY_BEEP = True
BEEP_FREQUENCY_HZ = 800
BEEP_DURATION_SEC = 0.20
BEEP_VOLUME = 0.50


# =========================================================
# Key settings
# =========================================================

KEY_SPACE = "space"
KEY_UP = "up"
KEY_DOWN = "down"
KEY_QUIT = "escape"

AFTER_SPACE_DELAY_SEC = 0.20


# =========================================================
# Experiment info
# =========================================================

def get_exp_info():
    exp_info = {
        "participant": f"{random.randint(0, 999999):06d}",
        "session": "001",
        "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    dlg = gui.DlgFromDict(
        dictionary=exp_info,
        title=EXP_NAME,
        sortKeys=False,
    )

    if not dlg.OK:
        core.quit()

    return exp_info


# =========================================================
# Window setup
# =========================================================

def make_monitor():
    mon = monitors.Monitor(
        name="runtime_monitor",
        width=MONITOR_WIDTH_CM,
        distance=VIEWING_DISTANCE_CM,
    )
    mon.setSizePix(SCREEN_RESOLUTION_PX)
    return mon


def make_stim_window():
    mon = make_monitor()

    if PILOT_MODE:
        return visual.Window(
            size=STIM_WINDOW_SIZE,
            fullscr=False,
            screen=0,
            pos=(40, 80),
            units="deg",
            color=BACKGROUND_COLOR,
            monitor=mon,
            allowGUI=True,
            name="stimulus_screen",
        )

    return visual.Window(
        fullscr=FULLSCREEN,
        screen=STIM_SCREEN_INDEX,
        units="deg",
        color=BACKGROUND_COLOR,
        monitor=mon,
        allowGUI=False,
        name="stimulus_screen",
    )


def make_probe_window():
    if PILOT_MODE:
        return visual.Window(
            size=PROBE_WINDOW_SIZE,
            fullscr=False,
            screen=0,
            pos=(860, 80),
            units="pix",
            color=BACKGROUND_COLOR,
            monitor="testMonitor",
            allowGUI=True,
            name="probe_screen",
        )

    return visual.Window(
        fullscr=FULLSCREEN,
        screen=PROBE_SCREEN_INDEX,
        units="pix",
        color=BACKGROUND_COLOR,
        monitor="testMonitor",
        allowGUI=False,
        name="probe_screen",
    )


# =========================================================
# Geometry / Setup Calculations (137-Degree Inward Wrap)
# =========================================================

def get_disc_position(eccentricity_deg):
    """
    Calculates the correct position for the LEFT monitor (Stimulus).
    Uses hardcoded physical targets based on 137° inward setup at 50 cm.
    """
    # 1. Hardcoded tape-measure targets from the central gap
    if eccentricity_deg == 8.0:
        target_cm = 7.2
    else: 
        target_cm = 25.3

    # 2. Subtract the 2.5 cm left monitor bezel
    dist_from_glass_edge = target_cm - STIM_BEZEL_WIDTH_CM

    # 3. Find center offset (Left monitor LED is on its right edge)
    # Distance from center (0) to the right edge is MONITOR_WIDTH_CM / 2.0
    x_cm_from_center = (MONITOR_WIDTH_CM / 2.0) - dist_from_glass_edge

    # 4. Convert physical offset back into PsychoPy's expected degrees
    x_pos_deg = np.degrees(np.arctan(x_cm_from_center / VIEWING_DISTANCE_CM))

    return x_pos_deg, 0.0


def get_probe_position_pix(eccentricity_deg): 
    """
    Calculates the correct position for the RIGHT monitor (Probe).
    Uses pure pixel math based on 137° inward setup at 50 cm.
    """
    # 1. Hardcoded tape-measure targets from the central gap
    if eccentricity_deg == 8.0:
        target_cm = 7.2
    else: 
        target_cm = 25.3

    # 2. Subtract the 0.5 cm right monitor bezel
    dist_from_glass_edge = target_cm - PROBE_BEZEL_WIDTH_CM
    
    # 3. Convert physical centimeters directly into raw pixels
    probe_screen_res_x = 1920.0 
    px_per_cm = probe_screen_res_x / PROBE_MONITOR_WIDTH_CM
    dist_pix = int(dist_from_glass_edge * px_per_cm)
    
    # 4. Offset from left edge (PsychoPy 0,0 is center, so left edge is -res/2)
    x_pos_pix = -(probe_screen_res_x / 2.0) + dist_pix
    
    return x_pos_pix, 0


def make_trials(n_trials):
    if not BALANCED_ECCENTRICITY:
        return [random.choice(ECCENTRICITIES_DEG) for _ in range(n_trials)]

    n_ecc = len(ECCENTRICITIES_DEG)
    base_repeats = n_trials // n_ecc
    remainder = n_trials % n_ecc

    trials = ECCENTRICITIES_DEG * base_repeats

    if remainder > 0:
        trials.extend(random.sample(ECCENTRICITIES_DEG, remainder))

    random.shuffle(trials)
    return trials


# =========================================================
# Drawing and sound
# =========================================================

def draw_fixation(fixation):
    if SHOW_FIXATION:
        fixation.draw()

def draw_blank_with_optional_fixation(win, fixation=None):
    win.color = BACKGROUND_COLOR
    if fixation is not None:
        draw_fixation(fixation)
    win.flip()

def blank_window_once(win):
    win.color = BACKGROUND_COLOR
    win.flip()

def play_beep():
    if not PLAY_BEEP:
        return
    try:
        if platform.system() == "Windows":
            winsound.Beep(BEEP_FREQUENCY_HZ, int(BEEP_DURATION_SEC * 1000))
        else:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
    except Exception as e:
        print(f"WARNING: Could not play beep: {e}")

def make_timestamp():
    return datetime.now().isoformat(timespec="milliseconds")

def show_message(win, kb, message, quit_windows=None):
    if quit_windows is None:
        quit_windows = [win]
    text = visual.TextStim(
        win, text=message, color=[1, 1, 1], height=0.5, wrapWidth=24, pos=(0, 0)
    )
    kb.clearEvents()
    while True:
        keys = kb.getKeys(keyList=[KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        for key in keys:
            if key.name == KEY_QUIT:
                safe_quit(quit_windows)
            if key.name == KEY_SPACE:
                play_beep()
                core.wait(AFTER_SPACE_DELAY_SEC)
                kb.clearEvents()
                return
        text.draw()
        win.flip()
        core.wait(0.002)

def show_final_message_once(stim_win, probe_win, kb, output_file):
    kb.clearEvents()
    if probe_win is not None:
        blank_window_once(probe_win)

    stim_win.color = BACKGROUND_COLOR
    stim_win.flip()

    final_text = visual.TextStim(
        win=stim_win,
        text=(f"Experiment finished.\n\nData saved to:\n{output_file}\n\nPress SPACE to exit."),
        color=[1, 1, 1], height=0.5, wrapWidth=24, pos=(0, 0)
    )
    while True:
        keys = kb.getKeys(keyList=[KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        for key in keys:
            if key.name == KEY_QUIT:
                safe_quit([stim_win, probe_win])
            if key.name == KEY_SPACE:
                play_beep()
                core.wait(AFTER_SPACE_DELAY_SEC)
                kb.clearEvents()
                return
        final_text.draw()
        stim_win.flip()
        core.wait(0.002)


# =========================================================
# Blurred probe generation
# =========================================================

def make_blurred_circle(res=256, radius_pix=60, blur_sigma=5.0, luminance=1.0):
    y, x = np.ogrid[:res, :res]
    cx, cy = res // 2, res // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    img = np.zeros((res, res), dtype=float)
    img[dist <= radius_pix] = luminance
    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma)
    img = np.clip(img, 0.0, 1.0)
    img = img * 2.0 - 1.0
    return img

def make_probe_stim(probe_win, blur_sigma, luminance=1.0, pos_pix=(0,0)): 
    img = make_blurred_circle(
        res=PROBE_IMAGE_RES_PIX, radius_pix=PROBE_RADIUS_PIX, 
        blur_sigma=blur_sigma, luminance=luminance
    )
    stim = visual.ImageStim(
        win=probe_win, image=img, size=(PROBE_SIZE_PIX, PROBE_SIZE_PIX), 
        pos=pos_pix, mask=None, units="pix"
    )
    return stim


# =========================================================
# Data handling
# =========================================================

def add_data_row(results, block, trial_index, eccentricity_deg, disc_x_deg, disc_y_deg, event_name, key="", rt="", blur_sigma="", luminance=""):
    results.append({
        "block": block, "trial_index": trial_index, "eccentricity_deg": eccentricity_deg,
        "disc_x_deg": disc_x_deg, "disc_y_deg": disc_y_deg, "event": event_name,
        "key": key, "rt": rt, "blur_sigma": blur_sigma, "luminance": luminance,
        "timestamp": make_timestamp(),
    })

def save_results(exp_info, results):
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    filename = data_dir / (f"{exp_info['participant']}_{EXP_NAME}_{exp_info['date']}.csv")
    fieldnames = [
        "participant", "session", "date", "block", "trial_index", "eccentricity_deg",
        "disc_x_deg", "disc_y_deg", "event", "key", "rt", "blur_sigma", "luminance", "timestamp"
    ]
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out = {"participant": exp_info["participant"], "session": exp_info["session"], "date": exp_info["date"], **row}
            writer.writerow(out)
    return filename


# =========================================================
# Block 1: key-logging task
# =========================================================

def wait_for_first_space_with_disc(stim_win, kb, disc, fixation):
    kb.clearEvents()
    while True:
        keys = kb.getKeys(keyList=[KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        for key in keys:
            if key.name == KEY_QUIT: safe_quit([stim_win])
            if key.name == KEY_SPACE:
                play_beep()
                return
        draw_fixation(fixation)
        disc.draw()
        stim_win.flip()
        core.wait(0.002)

def record_up_down_until_second_space(stim_win, kb, fixation, block_name, trial_index, eccentricity_deg, disc_x_deg, disc_y_deg, results):
    draw_blank_with_optional_fixation(stim_win, fixation)
    core.wait(AFTER_SPACE_DELAY_SEC)
    kb.clearEvents()
    trial_clock = core.Clock()
    add_data_row(results=results, block=block_name, trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="recording_started", rt="0.0000")

    while True:
        keys = kb.getKeys(keyList=[KEY_UP, KEY_DOWN, KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        for key in keys:
            rt = trial_clock.getTime()
            if key.name == KEY_QUIT: safe_quit([stim_win])
            elif key.name in [KEY_UP, KEY_DOWN]:
                add_data_row(results=results, block=block_name, trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="key_press", key=key.name, rt=f"{rt:.4f}")
            elif key.name == KEY_SPACE:
                play_beep()
                add_data_row(results=results, block=block_name, trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="trial_end", key=KEY_SPACE, rt=f"{rt:.4f}")
                core.wait(AFTER_SPACE_DELAY_SEC)
                kb.clearEvents()
                return
        draw_fixation(fixation)
        stim_win.flip()
        core.wait(0.002)

def run_block1_trial(stim_win, kb, fixation, trial_index, eccentricity_deg, results):
    disc_x_deg, disc_y_deg = get_disc_position(eccentricity_deg)
    disc = visual.Circle(win=stim_win, radius=DISC_RADIUS_DEG, fillColor=DISC_COLOR, lineColor=DISC_COLOR, pos=(disc_x_deg, disc_y_deg), units="deg")
    add_data_row(results=results, block="block1_keylog", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="disc_presented")
    wait_for_first_space_with_disc(stim_win=stim_win, kb=kb, disc=disc, fixation=fixation)
    record_up_down_until_second_space(stim_win=stim_win, kb=kb, fixation=fixation, block_name="block1_keylog", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, results=results)


# =========================================================
# Block 2: two-screen blur + luminance adjustment task
# =========================================================

def wait_for_space_with_disc_two_screens(stim_win, probe_win, kb, disc, fixation):
    kb.clearEvents()
    blank_window_once(probe_win)
    while True:
        keys = kb.getKeys(keyList=[KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        for key in keys:
            if key.name == KEY_QUIT: safe_quit([stim_win, probe_win])
            if key.name == KEY_SPACE:
                play_beep()
                return
        draw_fixation(fixation)
        disc.draw()
        stim_win.flip()
        core.wait(0.002)

def adjust_probe_blur_until_space(stim_win, probe_win, kb, fixation, trial_index, eccentricity_deg, disc_x_deg, disc_y_deg, results):
    blank_window_once(stim_win)
    core.wait(AFTER_SPACE_DELAY_SEC)
    kb.clearEvents()
    blur_sigma = PROBE_INITIAL_BLUR_SIGMA
    luminance = 1.0

    probe_x_pix, probe_y_pix = get_probe_position_pix(eccentricity_deg)
    probe_stim = make_probe_stim(probe_win=probe_win, blur_sigma=blur_sigma, luminance=luminance, pos_pix=(probe_x_pix, probe_y_pix))
    info_text = visual.TextStim(win=probe_win, text="Adjust blur\n\nUP = more blur\nDOWN = less blur\nSPACE = confirm blur", pos=(0, -330), color=[1, 1, 1], height=22, units="pix")
    blur_text = visual.TextStim(win=probe_win, text=f"Blur sigma = {blur_sigma:.1f}", pos=(0, 330), color=[1, 1, 1], height=26, units="pix")

    trial_clock = core.Clock()
    add_data_row(results=results, block="block2_blur_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="blur_adjustment_started", rt="0.0000", blur_sigma=f"{blur_sigma:.2f}", luminance=f"{luminance:.2f}")

    while True:
        keys = kb.getKeys(keyList=[KEY_UP, KEY_DOWN, KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        updated = False
        for key in keys:
            rt = trial_clock.getTime()
            if key.name == KEY_QUIT: safe_quit([stim_win, probe_win])
            elif key.name == KEY_UP:
                blur_sigma = min(PROBE_MAX_BLUR, blur_sigma + PROBE_BLUR_STEP)
                updated = True
                add_data_row(results=results, block="block2_blur_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="blur_changed", key=KEY_UP, rt=f"{rt:.4f}", blur_sigma=f"{blur_sigma:.2f}", luminance=f"{luminance:.2f}")
            elif key.name == KEY_DOWN:
                blur_sigma = max(PROBE_MIN_BLUR, blur_sigma - PROBE_BLUR_STEP)
                updated = True
                add_data_row(results=results, block="block2_blur_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="blur_changed", key=KEY_DOWN, rt=f"{rt:.4f}", blur_sigma=f"{blur_sigma:.2f}", luminance=f"{luminance:.2f}")
            elif key.name == KEY_SPACE:
                play_beep()
                add_data_row(results=results, block="block2_blur_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="final_blur_confirmed", key=KEY_SPACE, rt=f"{rt:.4f}", blur_sigma=f"{blur_sigma:.2f}", luminance=f"{luminance:.2f}")
                core.wait(AFTER_SPACE_DELAY_SEC)
                kb.clearEvents()
                return blur_sigma

        if updated:
            new_img = make_blurred_circle(res=PROBE_IMAGE_RES_PIX, radius_pix=PROBE_RADIUS_PIX, blur_sigma=blur_sigma, luminance=luminance)
            probe_stim.image = new_img
            blur_text.text = f"Blur sigma = {blur_sigma:.1f}"

        probe_stim.draw()
        if SHOW_PROBE_INSTRUCTIONS: info_text.draw()
        if SHOW_BLUR_VALUE_FOR_DEBUG: blur_text.draw()
        probe_win.flip()
        core.wait(0.002)

def adjust_probe_luminance_until_space(stim_win, probe_win, kb, fixation, trial_index, eccentricity_deg, disc_x_deg, disc_y_deg, final_blur_sigma, results):
    blank_window_once(stim_win)
    core.wait(AFTER_SPACE_DELAY_SEC)
    kb.clearEvents()
    luminance = PROBE_INITIAL_LUMINANCE
    luminance_phase_blur_sigma = LUMINANCE_ADJUSTMENT_BLUR_SIGMA

    probe_x_pix, probe_y_pix = get_probe_position_pix(eccentricity_deg)
    probe_stim = make_probe_stim(probe_win=probe_win, blur_sigma=luminance_phase_blur_sigma, luminance=luminance, pos_pix=(probe_x_pix, probe_y_pix))
    info_text = visual.TextStim(win=probe_win, text="Adjust luminance\n\nUP = brighter\nDOWN = darker\nSPACE = confirm", pos=(0, -330), color=[1, 1, 1], height=22, units="pix")
    luminance_text = visual.TextStim(win=probe_win, text=f"Luminance = {luminance:.2f}", pos=(0, 330), color=[1, 1, 1], height=26, units="pix")

    trial_clock = core.Clock()
    add_data_row(results=results, block="block2_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="luminance_adjustment_started", rt="0.0000", blur_sigma=f"{luminance_phase_blur_sigma:.2f}", luminance=f"{luminance:.2f}")

    while True:
        keys = kb.getKeys(keyList=[KEY_UP, KEY_DOWN, KEY_SPACE, KEY_QUIT], waitRelease=False, clear=True)
        updated = False
        for key in keys:
            rt = trial_clock.getTime()
            if key.name == KEY_QUIT: safe_quit([stim_win, probe_win])
            elif key.name == KEY_UP:
                luminance = min(PROBE_MAX_LUMINANCE, luminance + PROBE_LUMINANCE_STEP)
                updated = True
                add_data_row(results=results, block="block2_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="luminance_changed", key=KEY_UP, rt=f"{rt:.4f}", blur_sigma=f"{luminance_phase_blur_sigma:.2f}", luminance=f"{luminance:.2f}")
            elif key.name == KEY_DOWN:
                luminance = max(PROBE_MIN_LUMINANCE, luminance - PROBE_LUMINANCE_STEP)
                updated = True
                add_data_row(results=results, block="block2_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="luminance_changed", key=KEY_DOWN, rt=f"{rt:.4f}", blur_sigma=f"{luminance_phase_blur_sigma:.2f}", luminance=f"{luminance:.2f}")
            elif key.name == KEY_SPACE:
                play_beep()
                add_data_row(results=results, block="block2_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="final_luminance_confirmed", key=KEY_SPACE, rt=f"{rt:.4f}", blur_sigma=f"{luminance_phase_blur_sigma:.2f}", luminance=f"{luminance:.2f}")
                blank_window_once(probe_win)
                core.wait(AFTER_SPACE_DELAY_SEC)
                kb.clearEvents()
                return luminance

        if updated:
            new_img = make_blurred_circle(res=PROBE_IMAGE_RES_PIX, radius_pix=PROBE_RADIUS_PIX, blur_sigma=luminance_phase_blur_sigma, luminance=luminance)
            probe_stim.image = new_img
            luminance_text.text = f"Luminance = {luminance:.2f}"

        probe_stim.draw()
        if SHOW_PROBE_INSTRUCTIONS: info_text.draw()
        if SHOW_LUMINANCE_VALUE_FOR_DEBUG: luminance_text.draw()
        probe_win.flip()
        core.wait(0.002)

def run_block2_trial(stim_win, probe_win, kb, fixation, trial_index, eccentricity_deg, results):
    disc_x_deg, disc_y_deg = get_disc_position(eccentricity_deg)
    disc = visual.Circle(win=stim_win, radius=DISC_RADIUS_DEG, fillColor=DISC_COLOR, lineColor=DISC_COLOR, pos=(disc_x_deg, disc_y_deg), units="deg")
    add_data_row(results=results, block="block2_blur_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="disc_presented")
    wait_for_space_with_disc_two_screens(stim_win=stim_win, probe_win=probe_win, kb=kb, disc=disc, fixation=fixation)
    final_blur_sigma = adjust_probe_blur_until_space(stim_win=stim_win, probe_win=probe_win, kb=kb, fixation=fixation, trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, results=results)
    final_luminance = adjust_probe_luminance_until_space(stim_win=stim_win, probe_win=probe_win, kb=kb, fixation=fixation, trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, final_blur_sigma=final_blur_sigma, results=results)
    add_data_row(results=results, block="block2_blur_luminance_adjustment", trial_index=trial_index, eccentricity_deg=eccentricity_deg, disc_x_deg=disc_x_deg, disc_y_deg=disc_y_deg, event_name="trial_completed", blur_sigma=f"{final_blur_sigma:.2f}", luminance=f"{final_luminance:.2f}")


# =========================================================
# Quit helper
# =========================================================

def safe_quit(windows):
    for win in windows:
        if win is not None:
            try: win.close()
            except Exception: pass
    core.quit()


# =========================================================
# Main experiment
# =========================================================

def main():
    exp_info = get_exp_info()
    kb = keyboard.Keyboard()
    results = []
    stim_win = make_stim_window()
    fixation = visual.TextStim(win=stim_win, text="+", color=[1, 1, 1], height=FIXATION_SIZE_DEG, pos=(0, 0))
    probe_win = None

    try:
        draw_blank_with_optional_fixation(stim_win, fixation)
        show_message(
            stim_win, kb,
            ("Block 1\n\nA white disc will appear.\n\nPress SPACE to remove the disc and start recording.\n\nThen press UP or DOWN as needed.\n\nPress SPACE again to end the trial.\n\nPress SPACE to begin."),
            quit_windows=[stim_win],
        )

        block1_trials = make_trials(BLOCK1_N_TRIALS)
        for trial_index, eccentricity_deg in enumerate(block1_trials, start=1):
            run_block1_trial(stim_win=stim_win, kb=kb, fixation=fixation, trial_index=trial_index, eccentricity_deg=eccentricity_deg, results=results)

        show_message(
            stim_win, kb,
            ("Block 1 finished.\n\nThe next block will use two screens.\n\nPress SPACE to continue."),
            quit_windows=[stim_win],
        )

        probe_win = make_probe_window()
        draw_blank_with_optional_fixation(stim_win, fixation)
        blank_window_once(probe_win)

        show_message(
            stim_win, kb,
            ("Block 2\n\nA white disc will appear on the presentation screen.\n\nPress SPACE to remove the disc.\n\nThen adjust the probe disc on the other screen.\n\nDuring adjustment, the presentation screen will remain black.\n\nFirst adjustment:\nUP = more blur\nDOWN = less blur\nSPACE = confirm blur\n\nSecond adjustment:\nUP = brighter\nDOWN = darker\nSPACE = confirm luminance and go to the next trial\n\nPress SPACE to begin."),
            quit_windows=[stim_win, probe_win],
        )

        block2_trials = make_trials(BLOCK2_N_TRIALS)
        for trial_index, eccentricity_deg in enumerate(block2_trials, start=1):
            run_block2_trial(stim_win=stim_win, probe_win=probe_win, kb=kb, fixation=fixation, trial_index=trial_index, eccentricity_deg=eccentricity_deg, results=results)

        output_file = save_results(exp_info, results)
        show_final_message_once(stim_win=stim_win, probe_win=probe_win, kb=kb, output_file=output_file)

    finally:
        if probe_win is not None: probe_win.close()
        stim_win.close()
        core.quit()

if __name__ == "__main__":
    main()