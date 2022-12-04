from utils import Config
import json
arg_dict = json.load(open("../config/config.json"))
config = Config.from_dict(arg_dict)