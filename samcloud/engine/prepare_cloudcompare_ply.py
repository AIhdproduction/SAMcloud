"""Make an existing classified PLY expose its class IDs as a CloudCompare scalar field."""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="classified PLY with a 'class' vertex property")
    parser.add_argument("--output", required=True, help="output PLY with a 'scalar_class' vertex property")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        sys.exit("Input and output paths must be different.")

    with input_path.open("rb") as source:
        header = bytearray()
        while True:
            line = source.readline()
            if not line:
                sys.exit("Invalid PLY: end_header was not found.")
            header.extend(line)
            if line.rstrip(b"\r\n") == b"end_header":
                break

        old = b"property uchar class"
        new = b"property uchar scalar_class"
        if old not in header:
            sys.exit("The PLY header has no 'property uchar class' field.")
        header = header.replace(old, new, 1)

        with output_path.open("wb") as destination:
            destination.write(header)
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    print(f"Written CloudCompare-ready PLY: {output_path}")


if __name__ == "__main__":
    main()
