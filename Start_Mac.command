#!/bin/zsh
cd "$(dirname "$0")"
if command -v python3.13 >/dev/null 2>&1; then
  python3.13 setup_demo.py
elif command -v python3 >/dev/null 2>&1; then
  python3 setup_demo.py
else
  echo 'Install Python 3.13 from https://www.python.org/downloads/ and open this launcher again.'
  read 'reply?Press Enter to close...'
  exit 1
fi
if [ "$?" -ne 0 ]; then read 'reply?Press Enter to close...'; fi
