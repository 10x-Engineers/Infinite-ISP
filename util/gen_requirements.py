# Script for generating requirements.txt for this project
# Would'nt it be silly to run this file without python and pip

import subprocess

x = subprocess.check_output(["pip", "freeze"]).decode()

if ("pipreqs" not in x) :
    subprocess.run( ["pip", "install", "pipreqs"]  )

subprocess.run( ["pipreqs", "--force"]  )
