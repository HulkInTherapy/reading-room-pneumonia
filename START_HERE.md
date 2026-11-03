# Presenting tomorrow? Start here

You only need to follow one of these paths.

## 1. Presentation with no software setup

1. Open [GitHub Releases](https://github.com/HulkInTherapy/reading-room-pneumonia/releases/tag/v1.0.0).
2. Download **Presentation_Kit.zip** and extract it before leaving home.
3. Read **Friend_20_Minute_Guide.pdf** for the full story, then open **Presentation_8_Slides.pdf**. Use **Presenter_Notes.pdf** for the short read-aloud script.
4. Open **Offline_Demo.html** if you need the recorded backup. It works without Python or a network connection.

The offline page is clearly labelled as recorded model output. It does not run a new prediction.

## 2. Live model demonstration

1. Install **Python 3.13**, the version tested with this project.
2. Download this repository using **Code → Download ZIP**, then extract it. Or clone it with Git.
3. Double-click **Start_Windows.bat** on Windows or **Start_Mac.command** on Mac.
4. The first setup installs CPU libraries and downloads the 326 MiB model. Keep the terminal open and wait for the browser.
5. Select a sample and click **Analyze image**.

**Do the first setup before the presentation.** Subsequent launches reuse the model and environment. No GPU, SSH key, VPS access, API key or GitHub login is required. Linux CPU execution has been tested; Mac/Windows launcher scripts are supplied, but those operating systems have not been tested in this project.

If double-click does not work, open a terminal in the extracted repository folder and run:

```bash
python3.13 setup_demo.py
```

On Windows, use:

```powershell
py -3.13 setup_demo.py
```

The launcher opens a local address such as `http://127.0.0.1:8765`. It automatically chooses another free port if necessary. Keep that terminal open while presenting.

## What to show

1. Select a sample.
2. Click **Analyze image**.
3. Show **full X-ray → predicted lung mask → lung crop**.
4. Explain the final prediction and the known sample label.
5. Optionally show the four-condition comparison.

For uploaded images, lung segmentation runs now. Built-in samples use their prepared masks/crops, while classification still runs live. Individual scores are uncalibrated model outputs.

## What to say

> We classify public pediatric chest X-rays as normal or pneumonia. We tested lung cropping and a CNN/transformer combination using four matched comparisons. The final system reached 93.75% accuracy and 98.21% pneumonia recall on 624 test records. Validation and test showed different crop trends, so we report the limitation.

Read the [20-minute understanding guide](docs/Friend_20_Minute_Guide.pdf) first. All eight slides also have an exact script and simple Q&A in [Presenter_Notes.md](docs/Presenter_Notes.md).

## If something fails

- **Python version message:** install Python 3.13 and rerun the same launcher.
- **Download or package error:** check the connection and free disk space, then rerun; setup reuses completed work and verifies model hashes.
- **Terminal closed:** run the launcher again.
- **Port occupied:** the launcher selects a free port and prints its address.
- **No time to fix setup:** use **Offline_Demo.html** and explicitly call it recorded evidence.

Use the final documents in `docs/`. The old M1/M2/M3 material in `research/` is retained for history and may contain superseded proposals.
