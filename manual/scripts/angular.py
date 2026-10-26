"""Angular grains of the manual: the figures of the polygon and polyhedron examples, and the animation of the latter.

Run from the repository root with ``python manual/scripts/angular.py``. It runs ``examples/angular_grains.py``
and ``examples/angular_grains_3d.py``, writes their figures to ``manual/figures`` and moves the animation of the 3D
example, which a PDF cannot show, to ``examples/videos``, where the README shows it.
"""

import shutil

from common import FIGURES, ROOT, load_script

VIDEOS = ROOT / "examples" / "videos"


def main() -> None:
    """Run the two examples, writing their figures to the manual and the animation next to the examples."""
    VIDEOS.mkdir(exist_ok=True)
    print("--- angular_grains")
    load_script(ROOT / "examples" / "angular_grains.py").main(str(FIGURES))
    print("--- angular_grains_3d")
    load_script(ROOT / "examples" / "angular_grains_3d.py").main(str(FIGURES))
    shutil.move(FIGURES / "angular_grains_3d.gif", VIDEOS / "angular_grains_3d.gif")


if __name__ == "__main__":
    main()
