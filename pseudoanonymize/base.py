import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from faker import Faker
from openai import OpenAI

from pseudoanonymize.exceptions import MaxRetriesExceededException, UnparsableLLMOutputException
from pseudoanonymize.utils import flatten_replacement_dict

fake = Faker()


class BaseProcessor(ABC):
    def __init__(self, client: OpenAI):
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
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, f"{self.__class__.__name__}.log"))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _log_error(self, error_message: str, invalid_model_output: str) -> None:
        """
        Log an error message in case where the model fails to produce a valid output.
        """
        self.logger.error(f"Error: {error_message}\Model Output: {invalid_model_output}")

    def _parse_replacements(self, text: str, replacement_dict: Dict[Any, str]) -> str:
        """
        Using the replacement_dict, replace each key present in the dict with the corresponding value.
        """
        flat_dict = flatten_replacement_dict(replacement_dict)
        # Sort keys by length in descending order to replace longer strings first.
        # This avoids edge cases where a shorter key that is a subset of a longer key is replaced first, potentially interfering with subsequent replacements.
        sorted_keys = sorted(flat_dict.keys(), key=len, reverse=True)
        for key in sorted_keys:
            text = text.replace(key, str(flat_dict[key]))
        return text

    def _extract_and_parse_replacement_dict(self, combined_output: str) -> Dict[str, str]:
        replacement_dict_str = combined_output.strip()
        replacement_dict_str = replacement_dict_str[: replacement_dict_str.find("}") + 1]
        replacement_dict_str = replacement_dict_str[replacement_dict_str.find("{") :]
        try:
            replacement_dict = eval(replacement_dict_str)
        except (SyntaxError, ValueError) as e:
            error_message = f"Error: failed to parse replacement dictionary: {e}"
            self._log_error(error_message, combined_output)
            raise UnparsableLLMOutputException(error_message)
        return replacement_dict
