"""
Script for generating requirements.txt for this project
Would'nt it be silly to run this file without python and pip
"""

import subprocess

X = subprocess.check_output(["pip", "freeze"]).decode()

if "pipreqs" not in X:
    subprocess.run(["pip", "install", "pipreqs"], check=True)

subprocess.run(["pipreqs", "--force"], check=True)
