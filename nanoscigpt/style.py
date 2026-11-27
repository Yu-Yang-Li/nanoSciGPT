"""One look for every figure in the notebook and the docs: fonts, sizes, colours and a one-sentence headline."""
import unicodedata
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# One colour per role, used the same way in every figure.
INK = "#1d2630"        # text, observed data points, exact references
MUTED = "#7b8794"      # secondary text, grid, "starting point" references
MAIN = "#0b7285"       # the model or method the figure is about
SECOND = "#e8590c"     # the thing it is compared with
THIRD = "#5f3dc4"      # a third method, when there is one
GOOD, BAD = "#2f9e44", "#c92a2a"
PANEL = "#f6f8fa"      # light fill behind tables and missing values
SEQUENCE = "viridis"   # ordered quantities: rounds, sizes, frames, image intensity

# Noto Sans SC (SIL Open Font License), fetched only when the machine has no Chinese font at all.
FONT_URLS = ("https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
             "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf")
FONTS = ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "SimHei",
         "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback")


def chinese_font():
    """Name of an installed Chinese font. Without one, download Noto Sans SC once into Matplotlib's cache."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    found = [name for name in FONTS if name in installed]
    if found:
        return found[0]
    cached = Path(matplotlib.get_cachedir()) / "NotoSansSC-Regular.otf"
    for url in FONT_URLS:
        if cached.exists():
            break
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                cached.write_bytes(response.read())
        except OSError:
            continue
    if not cached.exists():
        return None
    font_manager.fontManager.addfont(cached)
    return font_manager.FontProperties(fname=cached).get_name()


def use():
    """Apply the style to every figure drawn afterwards. Returns the Chinese font, or None when none was found."""
    font = chinese_font()
    plt.rcParams.update({
        "font.family": [font] if font else ["DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#dfe4ea", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2.2, "lines.markersize": 7, "image.cmap": SEQUENCE,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.constrained_layout.use": True,
    })
    return font


def headline(fig, conclusion):
    """The figure's title is its conclusion in one sentence, left-aligned and wrapped to the figure width."""
    room = fig.get_figwidth() * 0.97 / 0.21  # a 15-point Chinese character is about 0.21 inch wide
    lines, width = [""], 0.0
    for character in conclusion:
        step = 1.0 if unicodedata.east_asian_width(character) in "WF" else 0.56  # Latin letters and digits are narrower
        if width + step > room:
            lines.append("")
            width = 0.0
        lines[-1] += character
        width += step
    fig.suptitle("\n".join(line.strip() for line in lines), x=0.01, ha="left", fontsize=15, fontweight="bold")


def plain(ax):
    """An axis that holds a picture or a table: no grid, no ticks."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def relative(value, reference):
    """"低 12%" or "高 3%": how a value compares with a reference, for headlines."""
    change = value / reference - 1
    return f"{'低' if change < 0 else '高'} {abs(change):.0%}"

