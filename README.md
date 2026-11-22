# MMWCAS Gesture Recognition

Jetson Nano viewer and recognizer for the TI MMWCAS burst firmware. This release combines two recognition models and two web views in one small package. 

## Defaults

- Model: **Context 17 standalone** (faster)
- Web view: **Compact Smart Room** (large capture counters, no plots)
- Capture: 28 continuous board-confirmed frames, then preprocessing and recognition

## Requirements

- Jetson Nano 4 GB
- TI MMWCAS running the compatible continuous-capture firmware
- Nano IPv4 `192.168.33.30` 
- Python 3.6 environment at `$HOME/venvs/lptf-nano-torch110`
- PyTorch 1.10, NumPy 1.19.5 and SciPy 1.4.1

The launcher checks the package files and the PyTorch version before starting.

For radar firmware and microSD configuration, see [`radar-card/README.md`](radar-card/README.md).

## Start

Extract the complete archive on the Nano, open a terminal in the extracted folder, and run:

```bash
bash run_dsp_gesture.sh
```

Open `http://127.0.0.1:8765` in Chromium. Click **Start Recognition**, perform one gesture until `CAPTURE 28/28` is shown, and wait for the result.

## Choose a model or page

```bash
# Default: fast Context 17 model + compact Smart Room
bash run_dsp_gesture.sh

# Full EF-CIR model + compact Smart Room
bash run_dsp_gesture.sh --model efcir

# Fast Context 17 model + four plots
bash run_dsp_gesture.sh --ui full

# Full EF-CIR model + four plots
bash run_dsp_gesture.sh --model efcir --ui full
```

| Option | Description |
| --- | --- |
| `--model context17` | Faster standalone 16-class model; raw finite XYZ points are interpolated to 120 points. |
| `--model efcir` | Full EF-CIR soft-routing model; origin removal and point filtering are applied before interpolation. |
| `--ui compact` | Default low-overhead Smart Room with large progress counters. |
| `--ui full` | Backed-up Smart Room page with RD, detection, X-Z and Y-Z plots. |


## Notes

- Do not start two viewers at the same time; only one process can use the radar and web ports.
- A partial or interrupted burst is discarded and retried automatically, up to three times.
- Saved recognition windows are written to `dsp-gesture-results/` only when **Save Result** is used in the full page.
- Stop the program with `Ctrl+C`.

## Quick checks

These checks do not require the radar:

```bash
python3 validation/test_board_progress.py
python3 validation/test_partial_capture_retry.py
```

## License

The original code and model weights in this project are licensed under the [MIT License](LICENSE).

Files under `radar-card/` contain Texas Instruments components and are distributed under their corresponding TI license and notices. The root MIT License does not replace or modify those terms.
