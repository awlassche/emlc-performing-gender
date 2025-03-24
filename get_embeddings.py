import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import ndjson

def chunk_speech(speech, chunk_size):
    """
    Splits a speech into chunks of a specified word length.

    :param speech: The full speech text.
    :param chunk_size: The maximum number of words per chunk.
    :return: A list of speech chunks.
    """
    words = speech.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_dataset(df, chunk_size):
    """
    Chunks speeches into smaller parts, filters out short chunks, and returns a DataFrame.

    :param df: The original DataFrame containing speeches.
    :param chunk_size: The chunk size to apply.
    :return: A filtered DataFrame with chunked speech.
    """
    # Group speeches per speaker/play/gender
    grouped = df.groupby(['speaker', 'gender', 'play'])['speech'].agg(' '.join).reset_index()

    chunked_data = []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Chunking grouped speeches"):
        chunks = chunk_speech(row['speech'], chunk_size)
        speaker_id_base = row['speaker'][:4]

        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 5:  # Optional: filter out very short chunks
                chunked_data.append({
                    'speaker': row['speaker'],
                    'play': row['play'],
                    'gender': row['gender'],
                    'speech_chunk': chunk,
                    'unique_id': f"{speaker_id_base}_{i + 1}"
                })

    return pd.DataFrame(chunked_data)

def embed_and_save(input_path, output_path, chunk_size=300, model_name="emanjavacas/GysBERT-v2"):
    # 1. Load raw data
    with open(input_path) as fin:
        data = ndjson.load(fin)
    df = pd.DataFrame(data)

    # 2. Chunk speeches
    df_chunked = process_dataset(df, chunk_size)

    # 3. Load embedding model
    model = SentenceTransformer(model_name)

    # 4. Encode with batching
    print("Generating embeddings...")
    embeddings = model.encode(
        df_chunked['speech_chunk'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    # 5. Attach and save
    df_chunked['embedding'] = embeddings.tolist()
    dataset = Dataset.from_pandas(df_chunked)
    dataset.save_to_disk(output_path)
    print(f"✅ Saved dataset with embeddings to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .ndjson file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save Hugging Face dataset")
    parser.add_argument("--chunk_size", type=int, default=300, help="Max number of words per chunk")
    parser.add_argument("--model", type=str, default="emanjavacas/GysBERT-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    embed_and_save(args.input, args.output, args.chunk_size, args.model)
