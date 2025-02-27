# config_manager.py
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("user.log"),  logging.StreamHandler()]
)

def get_executable_dir() -> Path:
    """Get executable directory (compatible with both development and packaged environments)"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.absolute()
    else:
        return Path(__file__).parent.absolute()

class ConfigHandler:
    # Define configuration structure with type annotations
    DEFAULT_CONFIG: Dict[str, Any] = {
        "user_settings": {
            "engine": "Google",
            "source_lang": "english",
            "target_lang": "chinese (simplified)",
            "model_dir": "vosk-model-small-en-us-0.15",
            "transcribe_device_index": 0,
            "monitor_position": [0, 0],
            "deepl_key": "",
            "ollama_url": "localhost:11434",
            "ollama_model": ""
        }
    }

    def __init__(self, config_name: str = "user_config.json"):
        self.config_path  = get_executable_dir() / config_name
        self.config  = deepcopy(self.DEFAULT_CONFIG)  # Prevent default value contamination
        self._convert_paths()

    def _convert_paths(self):
        """Convert relative paths to absolute paths in configuration"""
        model_path = self.config["user_settings"]["model_dir"]
        abs_path = (get_executable_dir() / model_path).resolve()
        self.config["user_settings"]["model_dir"]  = str(abs_path)

    def load_config(self) -> Dict[str, Any]:
        """Safely load configuration file with error handling"""
        try:
            if self.config_path.exists():
                with open(self.config_path,  'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    self._deep_merge(self.config,  user_config)
                    logging.info("Configuration  loaded successfully")
            else:
                self._ensure_config_dir()
                self.save_config()
        except json.JSONDecodeError as e:
            logging.error(f"Configuration  file format error: {e}, using default settings")
        except PermissionError as e:
            logging.error(f"Permission  denied accessing config file: {e}")
        except Exception as e:
            logging.error(f"Unexpected  error: {e}")
        return self.config

    def save_config(self, current_settings: dict = None):
        """Safe configuration persistence with validation"""
        try:
            self._ensure_config_dir()
            if current_settings:
                self._validate_settings(current_settings)
                self._deep_merge(self.config,  current_settings)

            with open(self.config_path,  'w', encoding='utf-8') as f:
                json.dump(self.config,  f, indent=4, ensure_ascii=False)
                logging.info("Configuration  saved successfully")
        except PermissionError as e:
            logging.error(f"Failed  to save configuration (permission denied): {e}")
        except Exception as e:
            logging.error(f"Error  saving configuration: {e}")

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Enhanced recursive merge algorithm for nested dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict):
                node = base.setdefault(key,  {})
                self._deep_merge(node, value)
            elif isinstance(value, list):
                base[key] = base.get(key,  []) + value
            else:
                base[key] = value

    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_path.parent.mkdir(parents=True,  exist_ok=True)

    def _validate_settings(self, settings: dict):
        """Basic configuration validation"""
        required_keys = {"engine", "source_lang", "target_lang"}
        if not required_keys.issubset(settings.get("user_settings",  {}).keys()):
            raise ValueError("Missing required configuration items")

        # Validate device index
        device_idx = settings["user_settings"].get("transcribe_device_index", -1)
        if device_idx < 0:
            raise ValueError("Invalid device index")