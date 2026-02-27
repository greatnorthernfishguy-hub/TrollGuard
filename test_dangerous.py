"""Test file with dangerous patterns for Cisco scanner simulation."""

import os
import subprocess
import base64
import pickle

# This should trigger threat detection
dangerous_code = "eval('print(\"bad\")')"
exec_code = "exec('import os; os.system(\"ls\")')"

# Shell injection
os.system("ls -la")

# Network access  
import urllib.request
urllib.request.urlopen("http://example.com")

# Base64 obfuscation
encoded = base64.b64decode("SGVsbG8=")

# Pickle deserialization
pickle.loads(b"data")

# Dynamic import
__import__("sys")