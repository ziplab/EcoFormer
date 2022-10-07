import json
import logging
import os
from datetime import datetime

from params import args

dt = datetime.now()
dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
_LOG_FMT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
_DATE_FMT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
logger = logging.getLogger("__main__")  # this is the global logger

log_path = os.path.join(args.output_dir, "all_logs.txt")
with open(os.path.join(args.output_dir, "args.json"), "w+") as f:
    json.dump(vars(args), f, indent=4)

fh = logging.FileHandler(log_path, "a+")
formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
fh.setFormatter(formatter)
logger.addHandler(fh)
