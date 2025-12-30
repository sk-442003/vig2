"""Remove known fake/synthetic example datasets from the repo.

This will only delete directories/files when --yes is provided to prevent accidents.
"""
import argparse
import shutil
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', help='Directories to remove', default=['data/face/example','data/face/augmented','data/audio/example','data/audio/augmented','data/text/example'])
    parser.add_argument('--yes', action='store_true', help='Actually delete (required)')
    args = parser.parse_args()

    for d in args.dirs:
        if os.path.exists(d):
            if args.yes:
                print('Removing', d)
                shutil.rmtree(d)
            else:
                print('Would remove', d)
        else:
            print('Not found:', d)
    if not args.yes:
        print('\nNo files deleted. Re-run with --yes to actually remove them.')

if __name__ == '__main__':
    main()
