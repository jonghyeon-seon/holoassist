import json
from collections import OrderedDict, Counter
import random
import argparse
import os
import shutil
import glob
import tqdm
import tarfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='demo microgesture')
    parser.add_argument('--data_version', '-v', default="0724",
                        help='data version')
    parser.add_argument('--data_path', '-d', default="/data/holoassist/",
                        help='data path')

    args = parser.parse_args()

    folder_list = os.listdir(args.data_path)
    print(folder_list)

    for folder_name in tqdm.tqdm(folder_list):
        if not (folder_name and folder_name[0] in ["z", "R", "Z", "r"]):
            continue

        base_path = os.path.join(args.data_path, folder_name, "Export_py")
        tar_path = os.path.join(base_path, "AhatDepth_synced.tar")
        extract_dir = os.path.join(base_path, "AhatDepth")

        # 1) .tar 풀기
        with tarfile.open(tar_path) as my_tar:
            my_tar.extractall(extract_dir)

        # 2) 중첩된 폴더에서 파일들을 꺼내오기
        nested_source = os.path.join(
            extract_dir,
            "mnt", "hl2data-westus2",
            "all-data-121922",
            folder_name,
            "Export_py", "AhatDepth"
        )

        for item in glob.glob(os.path.join(nested_source, "*")):
            filename = os.path.basename(item)
            dest = os.path.join(extract_dir, filename)

            # 이미 존재하면 건너뛰기
            if os.path.exists(dest):
                # print(f"[INFO] {filename} already exists, skipping.")
                continue

            try:
                shutil.move(item, extract_dir)
            except Exception as e:
                print(f"[WARN] {filename} 이동 실패: {e}")

        # 3) 중간 폴더 정리하기 (원하면 활성화)
        shutil.rmtree(os.path.join(extract_dir, "mnt"))
