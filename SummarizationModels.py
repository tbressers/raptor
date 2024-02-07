import logging
import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelWithLMHead
from abc import ABC, abstractmethod

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), reraise=True)
    def _attempt_summarize(self, context, max_tokens=150, stop_sequence=None):
        """
        This method attempts to generate a summary of the given context using the GPT-3 model. If it fails, it raises an exception which triggers a retry.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = openai.ChatCompletion.create(
          model=self.model,
          messages=[
                {"role": "system", "content": "You are a Summarizing Text Portal"},
                {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
            ],
          temperature=0
        )
        
        return response["choices"][0]['message']['content'].strip()

    def summarize(self, context, max_tokens=150, stop_sequence=None):
        """
        This method calls the _attempt_summarize method and handles any exceptions that may occur.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str or Exception: The generated summary if successful, or the exception if all retries fail.
        """
        try:
            return self._attempt_summarize(context, max_tokens=max_tokens, stop_sequence=stop_sequence)
        except Exception as e:
            print(e)
            return e

class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = openai.Completion.create(
                prompt=f"Write a summary of the following, including as many key details as possible: {context}. Summary:",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response["choices"][0]["text"].strip()

        except Exception as e:
            print(e)
            return ""
