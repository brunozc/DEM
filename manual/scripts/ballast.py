"""Ballast results of the manual: a sleeper on dry, half submerged and saturated ballast under cyclic loading.

Run from the repository root with ``python manual/scripts/ballast.py``. It runs ``examples/ballast_cyclic.py``, which
takes about twenty minutes on sixteen cores, and writes its figures to ``manual/figures``. The animations and the
ParaView files, which a PDF cannot show, are left to the example itself.
"""

from common import FIGURES, ROOT, load_script


def main() -> None:
    """Run the ballast example with its figures written to the manual."""
    example = load_script(ROOT / "examples" / "ballast_cyclic.py")
    example.main(str(FIGURES), animate=False, save_every_n_steps=0)


if __name__ == "__main__":
    main()
