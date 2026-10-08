# Input Video Directory

Place your test or benchmark video files here.

## Supported Formats
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`

## Usage Example
Place a video named `traffic.mp4` or `sample.mp4` into this directory, then run:

```bash
python main.py --source input/sample.mp4
```

To enable video recording and export the annotated tracking results to the `output/` directory:

```bash
python main.py --source input/sample.mp4 --save
```
