#!/usr/bin/env python
import json
import hashlib
from collections import OrderedDict
from pyexeggutor import (
    format_header,
    open_file_writer,
)

class LLMAnnotator(object):
    def __init__(
        self,
        api_key:str,
        organization_key:str,
        project_key:str,
        description:str=None,
    ):
        from openai import OpenAI
        self.description = description
        self.client = OpenAI(
            api_key=api_key,
            organization=organization_key,
            project=project_key,
        )
        self.history = OrderedDict()
        self.lookup = OrderedDict()

    @staticmethod
    def md5hash(string: str) -> str:
        """Compute a reproducible MD5 hash of a string."""
        return hashlib.md5(string.encode("utf-8")).hexdigest()

    def query(self, prompt:str, model="o3-mini", store=True):
        
        """
        Submits a prompt to the OpenAI API using the provided client and returns the response content as a string.

        Parameters:
            client: OpenAI client instance
                The initialized OpenAI client object.
            model: str
                The model to use for the completion (e.g., "o3-mini").
            prompt: str
                The user prompt to send to the OpenAI API.
            store: bool
                Whether to store the completion request (default is False).

        Returns:
            str: The content of the response from the OpenAI API.
        """
        id_hash = self.md5hash(prompt)
        if id_hash in self.history:
            return self.history[id_hash]
        else:
            # Submit the prompt to OpenAI
            completion = self.client.chat.completions.create(
                model=model,
                store=store,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Return the response content
            response = completion.choices[0].message
            content = response.content
            self.lookup[id_hash] = prompt
            self.history[id_hash] = content
            return content
        
    def to_json(self,filepath:str, sort_keys=True, indent=4):
        output = OrderedDict()
        for id_hash, prompt in self.lookup.items():
            content = self.history[id_hash]
            output[id_hash] = {"prompt":prompt, "content":content}
        with open_file_writer(filepath) as f:
            json.dump(output, f, sort_keys=sort_keys, indent=indent)
            
    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Description:{self.description})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + f"* number of queries: {len(self.history)}",
        ]

        return "\n".join(fields)