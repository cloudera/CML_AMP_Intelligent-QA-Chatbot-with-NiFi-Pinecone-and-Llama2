import subprocess

print(subprocess.run(["sh 1_session-install-deps/install_llama_deps.sh"], shell=True))