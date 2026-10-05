# forensically.py

A command-line port of the [Forensically](https://29a.ch/photo-forensics/) photo forensics toolkit. It needs only numpy and Pillow.

```bash
pip install -r requirements.txt
python3 forensically.py report photo.jpg          # everything -> photo_forensics/report.html + report.json
```

| Forensically tool | Command | Key options |
|---|---|---|
| Magnifier | `magnifier img -x X -y Y` | `--size --zoom --enhance {none,equalize,autocontrast,autocontrast-channel}` |
| Clone Detection | `clone img` | `--block --max-size --min-similarity --min-detail --min-cluster --min-distance --step --json out.json` |
| Error Level Analysis | `ela img` | `-q/--quality -s/--scale --opacity --equalize` |
| Noise Analysis | `noise img` | `-r/--radius -a/--amplitude --opacity --equalize` |
| Level Sweep | `sweep img` | `-p/--position -w/--width --animate N` (animated GIF) |
| Luminance Gradient | `gradient img` | `-i/--intensity --opacity --equalize` |
| Principal Component Analysis | `pca img` | `-c {1,2,3} -m {projection,difference,distance} --scaling {linear,abs,none} --invert --all` |
| Meta Data | `metadata img` | `--json --no-xmp` (IFD0, Exif, GPS, Interop, XMP, ICC, PNG text) |
| Geo Tags | `geo img` | `--json --open` (decimal coordinates plus OSM/Google Maps links) |
| Thumbnail Analysis | `thumbnail img` | `--extract thumb.jpg --scale --json` (side-by-side image and diff) |
| JPEG Analysis | `jpeg img` | `--json` (quant tables, quality estimate, standard vs custom table, subsampling, markers, data after EOI) |
| String Extraction | `strings img` | `-n --encoding {ascii,utf16,both} -g REGEX --offsets` |

By default, output images are written as `<image>_<tool>.png`. Use `-o` to choose a different path.

## Notes and limitations
- Clone detection finds copy-move by exact block shift. It uses lexicographically sorted block features, then a full-block RMS check, then clustering by shift vector. It does **not** detect clones that were rotated or scaled. Very flat areas are skipped (`--min-detail`). Lower `--min-detail` or `--min-cluster` to make it more sensitive. Raise `--max-size` to analyse at higher resolution (this is slower).
- EXIF orientation is not applied, so pixels are analysed exactly as stored.
- The quality estimate uses the IJG/libjpeg scaling formula. A "NON-standard table" result usually points to a camera's or an editor's own encoder.
- The algorithms copy what Forensically does, but they are not exact copies of its JavaScript. Output colours and scaling will differ slightly.
