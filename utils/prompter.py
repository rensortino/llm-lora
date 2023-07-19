"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, Optional


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt_old(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def generate_prompt(
        self,
        instruction: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """
        Generates a prompt for the given instruction, input and output using the specified prompt
        template.

        Args:
            instruction (Optional[str]):
                An optional string representing the instruction to be included in the prompt.
            input (Optional[str]):
                An optional string representing the input to be included in the prompt.
            output (Optional[str]):
                An optional string representing the output to be included in the prompt.

        Returns:
            str: The prompt string created using the specified prompt template.

        Raises:
            ValueError: If none of `instruction`, `input`, and `output` is defined.

        ## Example
        using ``

        {
        "instruction":
        },

        data_handler = DataHandler(tokenizer, "prompt_templates/medalpaca.json")
        prompt = data_hanlder.generate_prompt(
            instruction = "Provide a short answer to this medical question.",
            input = "What to expect if I have Aortic coarctation  (Outlook/Prognosis)?",
            output = (
                "The prognosis of aortic coarctation depends on whether balloon "
                "angioplasty and stenting or the surgery has been done or not."
            )
        )
        print(prompt)
        >>> Below is an instruction that describes a task, paired with an input that provides
            further context. Write a response that appropriately completes the request.

            ### Instruction:
            Provide a short answer to this medical question.

            ### Input:
            What to expect if I have Aortic coarctation  (Outlook/Prognosis)?

            ### Response:
            The prognosis of aortic coarctation depends on whether balloon angioplasty and
            stenting or the surgery has been done or not.
        """

        if not any([instruction, input, output]):
            raise ValueError("At least one of `instruction`, `input`, `output` should be defined")

        prompt = (
            f'{self.template["primer"]}'
            f'{self.template["instruction"]}{instruction or ""}'
            f'{self.template["input"]}{input or ""}'
            f'{self.template["output"]}{output or ""}'
        )

        return prompt

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
