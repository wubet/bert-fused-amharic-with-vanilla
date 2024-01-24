import os
import argparse
from translitration.transliteration import ethiopic2latin


def transliterate_text(original_file_path, transliterate_file_path):
    with open(original_file_path, encoding="utf8") as original_file:
        with open(transliterate_file_path, "w", encoding="utf8") as transliterate_file:
            amh_lines = original_file.readlines()
            for line in amh_lines:
                if len(line) > 2:
                    line = ethiopic2latin(line)
                    transliterate_file.write(line)


def main(args):
    base_path = os.path.abspath(os.getcwd())
    # Remove "transliteration" from the directory path for original_file_path
    original_path_dir = os.path.dirname(base_path)
    original_file_path = os.path.join(os.path.dirname(original_path_dir), args.original_filenames[0])
    transliterate_file_path = os.path.join(original_path_dir, args.transliterate_filenames[0])

    if not os.path.exists(transliterate_file_path) or os.path.getsize(transliterate_file_path) == 0:
        if os.path.exists(transliterate_file_path):
            with open(transliterate_file_path, 'r+') as f:
                f.truncate(0)
        transliterate_text(original_file_path, transliterate_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_filenames', nargs='+', required=True,
                        help='Names of files storing original language sequences.')
    parser.add_argument('--transliterate_filenames', nargs='+', required=True,
                        help='Names of files storing language transliteration sequences.')

    args = parser.parse_args()
    main(args)
