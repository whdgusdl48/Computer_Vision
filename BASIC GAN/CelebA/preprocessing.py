import os
import zipfile
import numpy as np

class preprocessor:
    target_root = None
    target_zip = None
    target_file = None
    def __init__(self):
        self.target_root = """./dataset"""
        self.target_zip = """archive.zip"""

    def run(self):
        state = self.unzip()

    def unzip(self):
        if not os.path.exists(self.target_root):
            raise Exception("No such dir name '{}'".format(self.target_root))
        img_folder = os.path.join(self.target_root, "img")
        if not os.path.exists(img_folder):
            target_zip = os.path.join(self.target_root, self.target_zip)
            if not os.path.exists(target_zip):
                raise Exception("No such file name '{}'".format(target_zip))

            os.mkdir(img_folder)
            zipfile.ZipFile(target_zip).extractall(img_folder)

    def ready_data(self):
        target_folder = os.path.join(self.target_root, "img")
a = preprocessor()
a.run()
