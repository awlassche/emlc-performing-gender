import xml.etree.ElementTree as ET
import os
import pandas as pd
import ndjson
import subprocess

def clone_repository():
    """
    Clone the Dutch Dracor repository containing TEI-encoded play texts.
    """
    subprocess.run(["git", "clone", "https://github.com/dracor-org/dutchdracor.git"], check=True)

def get_play_titles(directory):
    """
    Retrieve the list of play titles from the given directory.
    """
    return os.listdir(directory)

def parse_play(tree):
    """
    Parse a TEI-encoded play to extract speaker information and speeches.
    """
    root_play = tree.getroot()
    
    # Extract play metadata
    play_title = root_play.find(".//{http://www.tei-c.org/ns/1.0}titleStmt/{http://www.tei-c.org/ns/1.0}title").text
    play_id = root_play.get("{http://www.w3.org/XML/1998/namespace}id")
    
    # Create a dictionary of speakers
    speakers_dict = {}
    speaker_list = root_play.findall(".//{http://www.tei-c.org/ns/1.0}personGrp") + root_play.findall(".//{http://www.tei-c.org/ns/1.0}person")
    for speaker in speaker_list:
        speaker_id = speaker.get("{http://www.w3.org/XML/1998/namespace}id")
        name_element = speaker.find(".//{http://www.tei-c.org/ns/1.0}name") or speaker.find(".//{http://www.tei-c.org/ns/1.0}persName")
        name = name_element.text if name_element is not None else "Unknown"
        gender = speaker.get("sex", "Unknown")
        speakers_dict[speaker_id] = {"name": name, "gender": gender}
    
    # Collect speeches
    speech_elements = root_play.findall(".//{http://www.tei-c.org/ns/1.0}sp")
    speeches = []
    for speech in speech_elements:
        speaker_id = speech.get("who", "").strip("#")
        speaker_info = speakers_dict.get(speaker_id, {"name": "Unknown Speaker", "gender": "Unknown"})
        speaker_name = speaker_info["name"]
        speaker_gender = speaker_info["gender"]
        
        # Extract speech lines
        lines = speech.findall(".//{http://www.tei-c.org/ns/1.0}lb") + \
                speech.findall(".//{http://www.tei-c.org/ns/1.0}l") + \
                speech.findall(".//{http://www.tei-c.org/ns/1.0}s")
        lines = [line.text for line in lines if line.text is not None]
        
        speeches.append({'speaker': speaker_name, 'gender': speaker_gender, 'speech': lines, 'play': play_title, 'play_id': play_id})
    
    return speeches

def main():
    """
    Main function to process plays and save extracted data.
    """
    home_dir = 'dutchdracor/tei'
    
    # Clone the repository (uncomment if running for the first time)
    # clone_repository()
    
    # Retrieve the list of play titles
    list_of_titles = get_play_titles(home_dir)
    print(len(list_of_titles), "plays have been selected for analysis.")
    
    acts = []
    for title in list_of_titles:
        filename = os.path.join(home_dir, title)
        tree = ET.parse(filename)
        play_speeches = parse_play(tree)
        acts.extend(play_speeches)
    
    # Clean speech texts
    for act in acts:
        act['speech'] = [line for line in act['speech'] if line]
    
    # Convert to DataFrame and save as NDJSON
    df_acts = pd.DataFrame(acts)
    with open('speech_gender.ndjson', 'w') as fout:
        ndjson.dump(df_acts.to_dict('records'), fout)
    
    print("Processing completed. Data saved to speech_gender.ndjson.")

if __name__ == "__main__":
    main()