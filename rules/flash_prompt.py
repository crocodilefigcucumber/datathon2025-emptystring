import requests
import ujson as json
import os
import time
import concurrent.futures


def parse_boolean_from_string(value: str) -> bool:
    """
    Parses a string and returns its boolean value.
    Handles extra quotation marks and different capitalization.
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
    model: str = "models/gemini-1.5-flash-8b",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    """
    Sends a prompt to Gemini 1.5 Flash to check for discrepancies.
    Uses Google AI Studio API key (not OAuth2).
    """
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"{system_prompt.strip()}\n\n"
        f"Ground truth JSON:\n{json.dumps(ground_truth, indent=2)}\n\n"
        f"Free-form text:\n{summary_text.strip()}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }

    tick = time.time()
    response = requests.post(endpoint, headers=headers, json=payload)
    tock = time.time()
    print("Time taken for API call:", tock - tick)

    if response.status_code != 200:
        raise Exception(
            f"Gemini API request failed with status code {response.status_code}: {response.text}"
        )

    tick = time.time()
    data = response.json()
    tock = time.time()
    print("Time taken for JSON parsing:", tock - tick)

    try:
        output_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected response format: {data}")

    # Try to extract valid JSON from model output
    json_start = output_text.find("{")
    if json_start != -1:
        output_text = output_text[json_start:].replace("```", "")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nRaw model output:\n{output_text}")


def process_discrepancy(
    summary_text, ground_truth_data, discrepancy_system_prompt, api_key
):
    """
    Helper function to process a single discrepancy check.
    """
    try:
        discrepancies = send_full_discrepancy_request(
            discrepancy_system_prompt, ground_truth_data, summary_text, api_key
        )
        # Returns True if any discrepancy is found (i.e., sum of booleans > 0)
        return sum([entry for entry in discrepancies.values()]) > 0
    except Exception as e:
        print("An error occurred:", e)
        return False


def check_discrepancy_via_llm(folder_dir):
    """
    Check discrepancies between the client description and profile using LLM.
    This version parallelizes the API calls and returns False if a call takes longer than 1.1 seconds.

    Args:
        folder_dir (str): Path to the client folder.

    Returns:
        list: A list of booleans indicating whether a discrepancy was found for each pair.
    """
    tick = time.time()
    out = []
    desc_dict = {}
    client_dict = {}
    keys = ["higher_education", "employment_history", "financial"]

    if not os.path.isdir(folder_dir):
        raise Exception(f"Directory '{folder_dir}' not found.")

    # Load description and profile JSON files
    description_path = os.path.join(folder_dir, "client_description.json")
    with open(description_path, "r") as f:
        description = json.load(f)

    profile_path = os.path.join(folder_dir, "client_profile.json")
    with open(profile_path, "r") as f:
        profile = json.load(f)

    # Build description dictionary (exclude some keys)
    for key, value in description.items():
        if key not in ["Summary Note", "Family Background", "Client Summary"]:
            desc_dict[key] = value

    # Build client dictionary for specific keys
    for key, value in profile.items():
        if key in ["aum", "inheritance_details", "real_estate_details"]:
            client_dict["financial"] = [
                profile.get("aum", ""),
                profile.get("inheritance_details", ""),
                profile.get("real_estate_details", ""),
            ]
        elif key in keys:
            client_dict[key] = value

    # Load API key from file once
    with open("./gemini_api_key.txt", "r") as file:
        api_key = file.read().strip()

    # Define the discrepancy system prompt once
    discrepancy_system_prompt = """
    You are a discrepancy checker. Compare the ground truth JSON with a free-form text. For each key in the JSON (e.g., "secondary_school", "higher_education"), return a single boolean: "true" if there is any discrepancy between the ground truth and the free-form text for that key, and "false" if there is none. For nested objects or arrays, do not break them down further; treat them as a single unit. For empty arrays, return "false".
    Return only a JSON object with the same top-level keys as the ground truth and nothing else (DO NOT ADD ANY EXTRA KEYS, thank you). Do not include any additional text or explanations.
    """

    # Pair up the values from the two dictionaries
    pairs = list(zip(desc_dict.values(), client_dict.values()))

    # Parallelize API calls using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_discrepancy,
                summary_text,
                ground_truth_data,
                discrepancy_system_prompt,
                api_key,
            )
            for summary_text, ground_truth_data in pairs
        ]

        # Retrieve results with a timeout of 1.1 seconds per call
        for future in futures:
            try:
                result = future.result(timeout=1.1)
            except concurrent.futures.TimeoutError:
                print("A thread timed out. Returning False for that call.")
                result = False
            out.append(result)

    tock = time.time()
    print("Total time taken:", tock - tick)
    return out
