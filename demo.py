import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_IMAGE_PATH = "DataSet/Test/ResidentialTest/Residential_332.jpg"


def run_command(command):
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a quick LandUseVision evaluation and sample prediction."
    )
    parser.add_argument(
        "--image-path",
        default=DEFAULT_IMAGE_PATH,
        help="Sample image to classify after evaluation.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only run the sample prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    print("LandUseVision demo")

    if not args.skip_evaluation:
        run_command([sys.executable, "evaluate.py"])

    run_command([sys.executable, "predict.py", str(image_path)])


if __name__ == "__main__":
    main()
