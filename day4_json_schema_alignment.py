import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

candidate_data = """
Name: K Raviteja
Email: raviteja@example.com
Professional Role: GenAI Engineer
Primary Skill: Python and Generative AI
"""

schema_instruction = """
Convert the candidate information into a Schema.org Person JSON object.

The JSON must contain:
- @context
- @type
- name
- email
- jobTitle

Requirements:
1. Return only the JSON object.
2. Keep the entire JSON on one line.
3. Do not use markdown code blocks.
4. Do not add explanations or comments.
5. Use only information provided in the candidate data.
6. The result must be valid JSON.

Candidate information:
"""


def create_schema_response(candidate_info):

    try:
        result = api_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": schema_instruction
                },
                {
                    "role": "user",
                    "content": candidate_info
                }
            ]
        )

        generated_text = result.choices[0].message.content

        return generated_text.strip()

    except Exception as error:
        return f"Request failed: {error}"


def validate_schema_json(raw_output):

    try:
        parsed_data = json.loads(raw_output)

        return json.dumps(
            parsed_data,
            separators=(",", ":")
        )

    except json.JSONDecodeError:
        return "Validation failed: generated output is not valid JSON."


if __name__ == "__main__":

    print("Starting schema conversion...")

    api_output = create_schema_response(candidate_data)

    print("\nLLM Output:")
    print(api_output)

    final_json = validate_schema_json(api_output)

    print("\nValidated Schema JSON:")
    print(final_json)