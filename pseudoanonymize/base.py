import ast
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from faker import Faker

from .exceptions import (MaxRetriesExceededException,
                         UnparsableLLMOutputException)

fake = Faker()

class BaseProcessor(ABC):
    def __init__(self, client: Any):
        self.client = client
        self.logger = self._setup_logger()

    @abstractmethod
    def predict(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def retry_prediction(self, input_: Dict[str, Any], max_retries: int = 5) -> Optional[Dict[str, Any]]:
        retries = 0
        while retries < max_retries:
            try:
                prediction = self.predict(input_)
                return prediction
            except UnparsableLLMOutputException as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Failed to process item after {max_retries} attempts")
                    raise MaxRetriesExceededException(f"Failed to process item after {max_retries} attempts")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, f'{self.__class__.__name__}.log'))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _log_error(self, error_message: str, combined_output: str):
        self.logger.error(f"Error: {error_message}\nCombined Output: {combined_output}")

    def _parse_replacements(self, text, replacement_dict):
        cleaned_dict = dict()
        for key, replacement in replacement_dict.items():
            if isinstance(key, str):
                cleaned_dict[key] = replacement
            else:
                for key_instance in key:
                    cleaned_dict[key_instance] = replacement

        sorted_keys = sorted(cleaned_dict.keys(), key=len, reverse=True)
        for key in sorted_keys:
            text = text.replace(key, str(cleaned_dict[key]))
        return text

    def _extract_and_parse_replacement_dict(self, combined_output: str) -> Dict[str, str]:
        replacement_dict_str = combined_output.strip()
        replacement_dict_str = replacement_dict_str[:replacement_dict_str.find("}") + 1]
        replacement_dict_str = replacement_dict_str[replacement_dict_str.find("{"):]
        try:
            replacement_dict = eval(replacement_dict_str)
        except (SyntaxError, ValueError) as e:
            error_message = f"Error: failed to parse replacement dictionary: {e}"
            self._log_error(error_message, combined_output)
            raise UnparsableLLMOutputException(error_message)
        return replacement_dict