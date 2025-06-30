# Raspberry Pi Stereo Camera Calibration & Depth Extraction

If you're working with a Raspberry Pi stereo camera setup and want to extract depth information, stereo calibration is a must. This guide walks you through calibrating your stereo cameras using OpenCV — from capturing images to computing disparity and depth.

---

## Step 1: Prepare the Checkerboard

- **Pattern:** Use a printed checkerboard pattern with known square sizes.  
  Example: 9×6 checkerboard with 40 mm squares.
- **Download:** [Calibration Patterns Collection](https://markhedleyjones.com/projects/calibration-checkerboard-collection)
- **Tips:**
  - Print on matte paper (to avoid reflections).
  - Mount on a flat, rigid surface (like glass or cardboard).

---

## Step 2: Capture Calibration Images

Calibration works best with diverse images of the checkerboard in different positions, angles, and distances.

- **Script:** `stereo_calibration_collector.py` (captures synchronized image pairs)
- **Tips for Good Images:**
  - Ensure good lighting (no motion blur).
  - Checkerboard should be clearly visible in both left & right images.
  - Minimum: 30 image pairs (diverse poses).  
    *Example: 91 pairs captured using the script.*

### Vary Distance & Angles

| Distance (m)    | Poses per distance                                                                 |
|-----------------|------------------------------------------------------------------------------------|
| 8.2, 6, 5, 4, 3, 2, 1, 0.5 | - Centered (1–2 shots)<br>- Angled (1–2 shots: tilt/rotate)<br>- Off-centered (1–2: side/corner) |

### Script Controls

| Key | Function                      |
|-----|-------------------------------|
| s   | Start/resume saving (1/sec)   |
| p   | Pause saving                  |
| q   | Quit and save data            |

Images will be saved in the `calib/` folder as left01.png, right01.png... 

A CSV file logs the distances.

---

## Step 3: Run Stereo Calibration

- **Script:** `stereo_calibrate.py`
- **Function:** Finds checkerboard corners and estimates camera parameters.
- **Output:** `stereo_params.npz` (contains calibration data)

| Parameter   | Description                             |
|-------------|-----------------------------------------|
| mtxL, distL | Left camera intrinsics & distortion     |
| mtxR, distR | Right camera intrinsics & distortion    |
| R, T        | Rotation and translation between cameras|
| E, F        | Essential and fundamental matrices      |

---

## Step 4: Rectify Images & Compute Disparity

- **Script:** `stereo_rectify_and_disparity.py`
- **Function:** 
  - Load calibration file (`stereo_params.npz`)
  - Undistort & rectify images (`cv2.stereoRectify`)
  - Compute disparity map (`cv2.StereoSGBM_create`)
  - Save and visualize results

**OpenCV Functions Used:**

- `cv2.stereoRectify()`
- `cv2.initUndistortRectifyMap()`
- `cv2.remap()`
- `cv2.StereoSGBM_create()`

---

## Step 5 (Optional): Convert Disparity to Depth

Estimate depth with: 

$$
\text{depth} = \frac{\text{focalLength} \times \text{baseline}}{\text{disparity}}
$$

Where:
- **focalLength:** in pixels (from calibration)
- **baseline:** distance between cameras (meters)
- **disparity:** from disparity map

Alternatively, using the Q matrix:

$$
\text{depth} = \frac{Q[2,3]}{\text{disparity} + Q[3,2]}
$$

*Requirements:*
- Good calibration
- Accurate disparity
- Disparity > 0 (avoid divide-by-zero)

---

## Tools & Resources

| Item        | Source / Script                      |
|-------------|--------------------------------------|
| Checkerboard| [markhedleyjones.com](https://markhedleyjones.com/projects/calibration-checkerboard-collection) |
| Scripts     | `stereo_calibration_collector.py`<br>`stereo_calibrate.py`<br>`stereo_rectify_and_disparity.py`|
| Libraries   | OpenCV (`cv2`), NumPy                |

---

## Final Thoughts

Stereo calibration may seem complex, but once you walk through the process, it unlocks powerful 3D perception. Whether you're building a robot, mapping the world, or experimenting with depth, a calibrated stereo rig on the Raspberry Pi is a great starting point.
