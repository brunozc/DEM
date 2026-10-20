"""Three-dimensional examples of the manual: their figures for the manual and their animations for the README.

Run from the repository root with ``python manual/scripts/examples_3d.py``. It runs the four 3D examples, which take
a few minutes each, writes their figures to ``manual/figures`` and moves their animations, which a PDF cannot show,
to ``examples/videos``, where the README shows them.
"""

import shutil

from common import FIGURES, ROOT, load_script

EXAMPLES = ("square_footing_3d", "cone_penetration_3d", "hopper_discharge_3d", "upward_seepage_3d")
VIDEOS = ROOT / "examples" / "videos"


def main() -> None:
    """Run the 3D examples, writing their figures to the manual and their animations next to the examples."""
    VIDEOS.mkdir(exist_ok=True)
    for name in EXAMPLES:
        print(f"--- {name}")
        example = load_script(ROOT / "examples" / f"{name}.py")
        example.main(str(FIGURES))
        shutil.move(FIGURES / f"{name}.gif", VIDEOS / f"{name}.gif")


if __name__ == "__main__":
    main()
