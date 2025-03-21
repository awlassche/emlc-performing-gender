import pandas as pd
import numpy as np
import ndjson

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
import torch

# Function to count tokens in speech
def count_tokens(text):
    return len(text.split())

# Function to chunk speech into smaller parts
def chunk_speech(speech, chunk_size):
    """
    Splits a speech into chunks of a specified word length.

    :param speech: The full speech text.
    :param chunk_size: The maximum number of words per chunk.
    :return: A list of speech chunks.
    """
    words = speech.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to process the dataset with a given chunk size
def process_dataset(df, chunk_size):
    """
    Chunks speeches into smaller parts, filters out short chunks, and returns a DataFrame.

    :param df: The original DataFrame containing speeches.
    :param chunk_size: The chunk size to apply.
    :return: A filtered DataFrame with chunked speech.
    """
    chunked_data = []
    
    for _, row in df.iterrows():
        chunks = chunk_speech(row['speech'], chunk_size)
        speaker_id_base = row['speaker'][:4]  # First 4 characters of speaker's name
        
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                'speaker': row['speaker'],
                'play': row['play'],
                'gender': row['gender'],
                'speech_chunk': chunk,
                'unique_id': f"{speaker_id_base}_{i + 1}"
            })

    df_chunked = pd.DataFrame(chunked_data)

    # Filter out speech chunks that are too short
    df_filtered = df_chunked[df_chunked['speech_chunk'].apply(lambda x: len(x.split()) >= chunk_size)].reset_index(drop=True)

    return df_filtered

# Function to get embeddings
def get_embeddings_advanced(model_name, text_chunks, pooling="cls"):
    """
    Generates embeddings for a list of text chunks using a specified transformer model.

    :param model_name: Name of the model to use.
    :param text_chunks: List of text chunks.
    :param pooling: Pooling strategy ("cls" or "mean").
    :return: NumPy array of embeddings.
    """
    
    # Try to use SentenceTransformer for simplicity
    try:
        model = SentenceTransformer(model_name)
        return np.array([model.encode(chunk) for chunk in text_chunks])
    
    except Exception as e:
        print(f"⚠️ SentenceTransformer failed for {model_name}: {e}\nFalling back to AutoModel.")

    # Fallback to AutoModel if SentenceTransformer doesn't work
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS pooling
        elif pooling == "mean":
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
        else:
            raise ValueError("Pooling strategy must be 'cls' or 'mean'.")

        embeddings.append(emb)

    return np.array(embeddings)

# Function to perform clustering and evaluate with V-measure
def evaluate_clustering(embeddings, true_labels, n_clusters):
    """
    Performs K-Means clustering and evaluates clustering quality using the V-measure score.

    :param embeddings: NumPy array of embeddings.
    :param true_labels: List of ground truth labels.
    :param n_clusters: Number of clusters (unique labels).
    :return: V-measure score.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    return v_measure_score(true_labels, predicted_labels)


def main():
    """
    Main function to cluster speech texts and compute v-scores.
    """
    with open('data/speech_gender.ndjson') as fin:
        filename = ndjson.load(fin)
    df = pd.DataFrame(filename)

    df_grouped = df.groupby(['speaker', 'play']).agg({
        'gender': 'first',  # Keep the first occurrence of gender (assuming it is the same for a speaker in the same play)
        'speech': ' '.join  # Merge the speech text
    }).reset_index()

    df_sorted = df_grouped.copy()
    df_sorted['speech_length'] = df_sorted['speech'].apply(count_tokens)
    df_sorted = df_sorted.sort_values(by='speech_length', ascending=False).reset_index(drop=True)

    # Take the top 40 speakers
    df_top = df_sorted.iloc[:40]

    # Date
    today = '250321'

    # File to save results
    output_file = f"results/v_measure_results_{today}.txt"

    # List of chunk sizes to test
    chunk_sizes = [200, 300, 400]  # Modify as needed

    # List of models to test
    model_names = [
        "emanjavacas/GysBERT-v2",
        "DTAI-KULeuven/robbert-2023-dutch-large",
        "GroNLP/bert-base-dutch-cased",
        "xlm-roberta-large",
        "intfloat/multilingual-e5-large" 
    ]

    # Open file to write results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Chunk Size\tModel Name\tV-Measure Score\tNumber of Rows\n")  # Header row

        for chunk_size in chunk_sizes:
            print(f"\n🔍 Testing chunk size: {chunk_size}")
            
            # Process the dataset with the given chunk size
            df_filtered = process_dataset(df_top, chunk_size)
            num_rows = df_filtered.shape[0]

            # Get speech chunks and true labels
            text_chunks = df_filtered['speech_chunk'].tolist()
            true_labels = df_filtered['speaker'].tolist()
            n_clusters = len(set(true_labels))  # Number of unique speakers

            # Evaluate each model
            for model_name in model_names:
                print(f"Evaluating model: {model_name} (Chunk size: {chunk_size})")

                # Get embeddings for this model
                embeddings = get_embeddings_advanced(model_name, text_chunks)

                # Evaluate clustering using V-measure
                v_measure = evaluate_clustering(embeddings, true_labels, n_clusters)

                # Save results to file
                f.write(f"{chunk_size}\t{model_name}\t{v_measure:.4f}\t{num_rows}\n")

    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()

