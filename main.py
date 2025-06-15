import os
import sys

import streamlit.web.cli as stcli

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        # "--global.developmentMode=false",
        '--server.headless=true'
    ]
    sys.exit(stcli.main())
