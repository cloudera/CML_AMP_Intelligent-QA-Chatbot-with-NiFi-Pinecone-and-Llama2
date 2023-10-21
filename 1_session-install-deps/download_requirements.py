import subprocess

print(subprocess.run(["sh 1_session-install-deps/download_requirements.sh"], shell=True))