import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')


project_name = 'project'

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/config.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "main.py",
    "requirements.txt",
    "app.py",
    "Dockerfile",
    "params.yaml",
    "schema.yaml",
    "templates/index.html",
    "templates/result.html",
    "config/config.yaml",
    "setup.py",
    "test.py"
    ]

for filepaths in list_of_files:
    filepath = Path(filepaths)
    file_dir,file_name = os.path.split(filepath)

    if file_dir != '':
        os.makedirs(file_dir,exist_ok=True)

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Created {filepath}")
    else:
        logging.info(f"{filepath} already exists")
