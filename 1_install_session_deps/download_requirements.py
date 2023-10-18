import subprocess

print(subprocess.run(["sh 1_install_session_deps/download_requirements.sh"], shell=True))