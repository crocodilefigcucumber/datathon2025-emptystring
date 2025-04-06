import requests
import json
import os


def parse_boolean_from_string(value: str) -> bool:
    """
    Parses a string and returns its boolean value.
    Handles extra quotation marks and different capitalization.

    Args:
        value (str): The string to parse.

    Returns:
        bool: The parsed boolean value. Defaults to False if in doubt.
    """
    normalized_value = value.strip().strip('"').strip("'").lower()
    if normalized_value in ["true", "yes", "1"]:
        return True
    elif normalized_value in ["false", "no", "0"]:
        return False
    else:
        return False


def send_full_discrepancy_request(
    system_prompt: str,
    ground_truth: dict,
    summary_text: str,
    api_key: str,
    model: str = "models/gemini-1.5-flash-latest",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    """
    Sends a prompt to Gemini 1.5 Flash to check for discrepancies between ground truth and free-form text.

    Args:
        system_prompt (str): The instruction for the model.
        ground_truth (dict): Structured ground truth data.
        summary_text (str): The unstructured user summary.
        api_key (str): Gemini API key.
        model (str): Gemini model endpoint.
        max_tokens (int): Maximum response tokens.
        temperature (float): Randomness factor.
        top_p (float): Top-p sampling control.

    Returns:
        dict: JSON result from the model.
    """
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prompt = (
        f"{system_prompt.strip()}\n\n"
        f"Ground truth JSON:\n{json.dumps(ground_truth, indent=2)}\n\n"
        f"Free-form text:\n{summary_text.strip()}"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens
        }
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(
            f"Gemini API request failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    try:
        output_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected response format: {data}")

    # Try to extract valid JSON from model output
    json_start = output_text.find("{")
    if json_start != -1:
        output_text = output_text[json_start:]

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nRaw model output:\n{output_text}")


def check_discrepancy_via_llm(folder_dir):
    """
    Check discrepancies between the client description and profile using LLM.
    Args:
        folder_dir (str): Path to the client folder.
    """

    out = []
    # create dictionary which compares the values from the two files
    desc_dict = {}
    # {description_field1 : description1, ... }
    client_dict = {}
    # {key1_of_desc_dict: extract(Education), ...}
    keys = ["higher_education", "employment_history", "financial"]

    if not os.path.isdir(folder_dir):
        raise Exception(f"Directory '{folder_dir}' not found.")

    description_path = os.path.join(folder_dir, "client_description.json")

    with open(description_path, "r") as f:
        description = json.load(f)

    profile_path = os.path.join(folder_dir, "client_profile.json")
    with open(profile_path, "r") as f:
        profile = json.load(f)

    for key, value in description.items():
        if key not in ["Summary Note", "Family Background", "Client Summary"]:
            desc_dict[key] = value

    for key, value in profile.items():
        if key in ["aum", "inheritance_details", "real_estate_details"]:
            client_dict["financial"] = [
                profile.get("aum", ""),
                profile.get("inheritance_details", ""),
                profile.get("real_estate_details", ""),
            ]
        elif key in keys:
            client_dict[key] = value

    with open("./openai_key.txt", "r") as file:
        api_key = file.read().strip()

    for value1, value2 in zip(desc_dict.values(), client_dict.values()):
        ground_truth_data = value2
        summary_text = value1

        # Example ground truth JSON data
        # ground_truth_data = {
        # "secondary_school": {"name": "Ålborg Katedralskole", "graduation_year": 2022},
        # "higher_education": [],
        # }

        # Example free-form text
        # summary_text = "In 2009, Jørgensen graduated from Ålborg Katedralskole with a secondary school diploma.\n"

        # System prompt instructing the LLM to fill out discrepancy info for each entry.
        discrepancy_system_prompt = """
        You are a discrepancy checker. Compare the ground truth JSON with a free-form text. For each key in the JSON (e.g., "secondary_school", "higher_education"), return a single boolean: "true" if there is any discrepancy between the ground truth and the free-form text for that key, and "false" if there is none. For nested objects or arrays, do not break them down further; treat them as a single unit. For empty arrays, return "false".
        Return only a JSON object with the same top-level keys as the ground truth and nothing else (DO NOT ADD ANY EXTRA KEYS, thank you). Do not include any additional text or explanations.
        """

        # Load API key from file
        try:
            discrepancies = send_full_discrepancy_request(
                discrepancy_system_prompt, ground_truth_data, summary_text, api_key
            )
            out.append(sum([entry for entry in discrepancies.values()]) > 0)
        except Exception as e:
            print("An error occurred:", e)
            out.append(False)

    return out
