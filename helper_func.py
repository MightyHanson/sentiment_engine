import os
import datetime
from TextObject import TextObject
def concatenate_txt_files(directory_path, output_filename):
    output_file_path = os.path.join(directory_path, output_filename)

    # If the output file already exists, delete it
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Get all the .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

    combined_content = []

    # Concatenate the contents of each file into the combined_content list
    for fname in txt_files:
        with open(os.path.join(directory_path, fname), encoding='utf-8') as infile:
            for line in infile:
                # Only add non-empty lines to the combined content
                if line.strip():
                    combined_content.append(line)

    # Write the combined content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(combined_content)

    print(f"All .txt files from {directory_path} have been concatenated into {output_file_path}")


def extract_text_content(filename) -> list:
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Splitting the content by the delimiter
    cases = content.split("--------------------------------------------------")

    # Extracting "Text Content" from each case
    text_contents = []
    for case in cases:
        # Splitting each case into lines
        lines = case.strip().split("\n")
        for line in lines:
            if line.startswith("Text Content:"):
                text_contents.append(line[len("Text Content: "):])
                break

    return text_contents

import ast

def read_text_objects_from_txt(filepath):
    text_objects = []
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Source: "):
                source = lines[i].split("Source: ")[1].strip()
                i += 1
                text_content = lines[i].split("Text Content: ")[1].strip()
                i += 1
                metadata_str = lines[i].split("Metadata: ")[1].strip()
                # Convert the string representation of the dictionary back to an actual dictionary
                metadata = ast.literal_eval(metadata_str)
                text_objects.append(TextObject(source, text_content, metadata))
            i += 1
    return text_objects

