import subprocess
import os

if os.getenv("VECTOR_DB") == "CHROMA":
    print(subprocess.run(["sh 1_session-install-deps/setup-chroma.sh"], shell=True))