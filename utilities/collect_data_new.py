import pandas as pd
import pathlib
import os
import json
import numpy as np
import sys
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def load_or_create(filename: str, rules: list, embedding: int, mode: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        data = collect_enriched(mode=mode, filename=filename, rules=rules, embedding=embedding)
        print("no save found, collecting data")
    else:
        data = pd.read_csv(filename)
        print("file loaded, checking completeness")
        old_data = True
        for rule in rules:
            if rule.__name__ not in data.columns:
                old_data = False
        if embedding > 0:
            for i in range(embedding):
                if f"pc_{i+1}" not in data.columns:
                    old_data = False
        if not old_data:
            print("data incomplete, collecting data")
            data = collect_enriched(mode=mode, filename=filename, rules=rules, embedding=embedding)
    return data


def collect_enriched(mode: str, filename: str, rules: list, embedding: bool) -> pd.DataFrame:
    n_rules = len(rules)

    if mode in ["train", "test", "val"]:
        dataset_path = "data/"
        split_path = "splits/" + mode + "_split.csv"

        clients = pd.read_csv(split_path)["file_path"].tolist()
    else:
        dataset_path = "final/"
        clients = os.listdir(dataset_path)
        clients = [client for client in clients if "zip" not in client]

    # Read client list from CSV and sort them (this list is used for both PCA and enriched features)
    
    clients = sorted(clients)
    n_clients = len(clients)

    if embedding:
        # ------------------------------------------------------------
        # 1. Compute PCA embeddings for all clients (before main loop)
        # ------------------------------------------------------------
        # Load the embedding model once
        model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        embedder = SentenceTransformer(model_name, device="cpu")  # or "cuda" if available

        # Define maximum number of words per section chunk (to respect context window limits)
        max_section_words = 100

        all_client_embeddings = []
        for client in tqdm(clients, desc="Computing PCA embeddings"):
            client_folder = os.path.join(dataset_path, client)
            # We extract text only from keys starting with "client_description"
            text_agg = ""
            documents = list(pathlib.Path(client_folder).glob("*.json"))
            json_file = os.path.join(client_folder, "client_description.json")
            try:
                with open(json_file, "r") as file:
                    file_data = json.load(file)
                for key, value in file_data.items():
                    text_agg += f"{key}: {str(value).strip()}\n\n"
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")
            # Split the aggregated text into sections (using double-newline as the delimiter)
            sections = [sec for sec in text_agg.strip().split("\n\n") if sec.strip()]
            # Further split a section if it exceeds max_section_words
            chunks = []
            for section in sections:
                words = section.split()
                if len(words) > max_section_words:
                    for i in range(0, len(words), max_section_words):
                        chunk = " ".join(words[i:i + max_section_words])
                        chunks.append(chunk)
                else:
                    chunks.append(section)
            # Get embeddings for each chunk and mean-pool them to get the client embedding
            if chunks:
                chunk_embeddings = embedder.encode(chunks)
                chunk_embeddings = np.array(chunk_embeddings)
                client_embedding = np.mean(chunk_embeddings, axis=0)
            else:
                # If no text found, create a zero vector with the same dimension as the model's embedding dimension
                client_embedding = np.zeros(embedder.get_sentence_embedding_dimension())
            all_client_embeddings.append(client_embedding)

        all_client_embeddings = np.array(all_client_embeddings)
        k = 5  # number of principal components
        pca = PCA(n_components=k)
        reduced_embeddings = pca.fit_transform(all_client_embeddings)
        df_pca = pd.DataFrame(
            reduced_embeddings,
            columns=[f"pc_{i+1}" for i in range(k)],
            index=clients
        ).reset_index().rename(columns={"index": "client_id"})
        
    # ------------------------------------------------------------
    # 2. Main loop: Collect enriched features (minimal changes from original)
    # ------------------------------------------------------------
    enriched = pd.DataFrame(data=np.zeros((n_clients, n_rules)))
    enriched.columns = [rule.__name__ for rule in rules]
    all_data = []

    i = 0
    for client in clients:
        client_folder = os.path.join(dataset_path, client)
        client_data = {"folder_name": client}  # Initialize with folder name

        # Get all JSON documents in the client's folder
        documents = list(pathlib.Path(client_folder).glob("*.json"))

        if not documents:
            print(f"No JSON files found in folder: {client}")
            all_data.append(client_data)
            i += 1
            continue

        # Extract features using the provided rules
        for rule in rules:
            name = rule.__name__
            accept, comment = rule(client_folder)
            enriched.loc[i, name] = float(accept)

        # Process each JSON file to collect client data
        for json_file in documents:
            try:
                with open(json_file, "r") as file:
                    file_data = json.load(file)
                    file_prefix = json_file.stem
                    for key, value in file_data.items():
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                client_data[f"{file_prefix}_{key}_{nested_key}"] = nested_value
                        else:
                            client_data[f"{file_prefix}_{key}"] = value
            except json.JSONDecodeError:
                print(f"Invalid JSON format in file: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")
        all_data.append(client_data)
        i += 1

    # Convert list of dictionaries to DataFrame (client data)
    df = pd.DataFrame(all_data)

    # Merge PCA results into the DataFrame based on the client id (here stored in "folder_name")
    if embedding:
        df_final = df.merge(df_pca, left_on="folder_name", right_on="client_id", how="left")
    else:
        df_final = df

    # Concatenate the enriched features (rule outputs)
    df_final = pd.concat([df_final, enriched], axis=1)
    
    df_final.to_csv(filename, index=False)
    return df_final

def collect_to_csv(clients: list, filename: str) -> pd.DataFrame:
    all_data = []

    for client in clients:
        client_folder = os.path.join("./data", client)
        client_data = {"folder_name": client}  # Initialize with folder name

        # Get all json documents
        documents = list(pathlib.Path(client_folder).glob("*.json"))

        if not documents:
            print(f"No JSON files found in folder: {client}")
            all_data.append(client_data)
            continue

        # Process each JSON file in the folder
        for json_file in documents:
            try:
                # Read the JSON file
                with open(json_file, "r") as file:
                    file_data = json.load(file)
                    file_prefix = json_file.stem
                    for key, value in file_data.items():
                        # Flatten nested dictionaries (optional - comment out if not needed)
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                client_data[f"{file_prefix}_{key}_{nested_key}"] = nested_value
                        else:
                            client_data[f"{file_prefix}_{key}"] = value
            except json.JSONDecodeError:
                print(f"Invalid JSON format in file: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")

        # Add this folder's data to the list
        all_data.append(client_data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    return df
