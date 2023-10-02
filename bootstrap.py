import subprocess

ret = subprocess.run(["sh", "cdsw-build.sh"])
print("The exit code for command 2 was: %d" % ret.returncode)