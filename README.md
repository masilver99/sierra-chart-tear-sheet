# Sierra Chart Tear Sheet

This project aims to provide in depth statistics from an exported Sierra Chart Trade Actvity Log.  This log provide rich data on every trade, including profit targets, stop losses, etc.

The motivation for this is based on the great quant stats tear sheet developed here: https://github.com/ranaroussi/quantstats

# Usage

Download the repo into a directory and from outside the directory run:

```
python -m tearsheet --input [YourExportFileName.txt] --output report.html
```
You can then double-click on report.html to view your tear sheet report.

This will generate an html report similar to this [sample report](https://masilver99.github.io/sierra-chart-tear-sheet/report.html)

