import requests
import json
import os 


def send_full_discrepancy_request(
    system_prompt: str,
    ground_truth: dict,
    summary_text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: list = ["<|im_end|>"],
) -> dict:
    """
    Sends a prompt to the LLM (OpenAI GPT-4) to check for discrepancies between the provided ground truth JSON
    and the free-form summary text. The LLM is instructed to return a JSON object with the same structure
    as the ground truth, where each value is replaced by a boolean indicating a discrepancy (true) or not (false).

    Args:
        system_prompt (str): The system prompt with instructions for discrepancy checking.
        ground_truth (dict): The ground truth JSON data.
        summary_text (str): The free-form text (e.g., wealth summary).
        api_key (str): API key for authentication.
        model (str): The model to be used.
        max_tokens (int): Maximum tokens allowed in the response.
        temperature (float): Temperature for controlling randomness.
        top_p (float): Top-p sampling parameter.
        stop (list): List of stop tokens.

    Returns:
        dict: A JSON object with the same structure as ground_truth where each entry's value is a boolean.
    """
    # Create messages for the Chat API: system instruction and user content.
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Ground truth JSON:\n{json.dumps(ground_truth, indent=2)}\n\nFree-form text:\n{summary_text}\n",
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
    )

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

    resp_json = response.json()
    if "choices" not in resp_json:
        raise ValueError("Invalid response format. 'choices' not found in response.")

    # Extract the assistant's message
    raw_output = resp_json["choices"][0]["message"]["content"].strip()

    # Remove markdown formatting if present
    if raw_output.startswith("```") and raw_output.endswith("```"):
        raw_output = raw_output.strip("```").strip()

    # Remove any text before the first '{' character
    json_start = raw_output.find("{")
    if json_start != -1:
        raw_output = raw_output[json_start:]

    try:
        parsed = json.loads(raw_output)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nRaw model output:\n{raw_output}")


def :

    #create dictionary which compares the values from the two files
    desc_dict = {}
    #{description_field1 : description1, ... }
    client_dict = {}
    #{key1_of_desc_dict: extract(Education), ...}
    keys = ["higher_education", "employment_history", "financial"]

    if not os.path.isdir(folder_dir):
        raise Exception(f"Directory '{folder_dir}' not found.")
    
    description_path = os.path.join(folder_dir, "client_description.json")

    with open(description_path, "r") as f:
        description = json.load(f)
    

    profile_path = os.path.join(folder_dir, "client_profile.json")
    with open(profile_path, "r") as f:
        profile = json.load(f)
    
    for i, value in zip(len(keys),description.values()[1:]):
        desc_dict[keys[i]] = value
    
    for key, value in profile.keys():
        if key in ["aum", "inheritance_details", "real_estate_details"]:
            client_dict["financial"] = [profile.get("aum", ""), profile.get("inheritance_details", ""), profile.get("real_estate_details", "")]
        else:
            client_dict[key] = value


with open("./openai_key.txt", "r") as file:
    api_key = file.read().strip()



for value1, value2 in zip(desc_dict.values(), client_dict.values()):
    ground_truth_data = value2
    summary_text = value1

    # Example ground truth JSON data
    #ground_truth_data = {
        #"secondary_school": {"name": "Ålborg Katedralskole", "graduation_year": 2022},
       # "higher_education": [],
    #}

    # Example free-form text
    #summary_text = "In 2009, Jørgensen graduated from Ålborg Katedralskole with a secondary school diploma.\n"

    # System prompt instructing the LLM to fill out discrepancy info for each entry.
    discrepancy_system_prompt = """
    You are a discrepancy checker. Compare the ground truth JSON with a free-form text. For each key in the JSON (e.g., "secondary_school", "higher_education"), return a single boolean: "true" if there is any discrepancy between the ground truth and the free-form text for that key, and "false" if there is none. For nested objects or arrays, do not break them down further; treat them as a single unit. For empty arrays, return "false".
    Return only a JSON object with the same top-level keys as the ground truth and nothing else. Do not include any additional text or explanations.
    """

    # Load API key from file

    try:
        discrepancies = send_full_discrepancy_request(
            discrepancy_system_prompt, ground_truth_data, summary_text, api_key
        )
        print("Discrepancy JSON:")
        print(json.dumps(discrepancies, indent=2))
    except Exception as e:
        print("An error occurred:", e)
