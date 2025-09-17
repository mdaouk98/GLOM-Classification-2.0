# src/validate_config.py
import argparse
from config import load_config

def validate(config_path: str):
    try:
        config = load_config(config_path)
        print("Configuration is valid.")
    except Exception as e:
        print(f"Configuration validation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file.')
    args = parser.parse_args()
    validate(args.config)
