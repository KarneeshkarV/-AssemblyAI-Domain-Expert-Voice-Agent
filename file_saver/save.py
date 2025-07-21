import argparse
import datetime
import os


def save_text_to_file(text: str, filename: str = None, output_dir: str = "output"):
    """
    Saves the given text to a file.

    Args:
        text (str): The text to save.
        filename (str, optional): The desired filename. If not provided, a timestamp-based name is generated.
        output_dir (str, optional): The directory to save the file in. Defaults to "output".
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine the filename
    if filename:
        if not filename.endswith(".txt"):
            filename += ".txt"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.txt"

    file_path = os.path.join(output_dir, filename)

    # Write the text to the file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully saved text to {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


def read_all_files_in_folder(folder_path: str) -> str:
    """
    Reads all files from a folder and returns their concatenated content.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        str: The combined content of all files, separated by a newline.
    """
    combined_content = ""
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at {folder_path}")
        return ""
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    combined_content += f.read() + "\n"
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
    return combined_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save or read text to/from a file.")
    parser.add_argument("--save", action="store_true", help="Save text to a file.")
    parser.add_argument("--read", action="store_true", help="Read all files from a folder.")
    parser.add_argument("--text", help="The text content to save.")
    parser.add_argument("--filename", help="Optional: the name of the file to save the text to.")
    parser.add_argument("--folder", help="The folder to read files from.")

    args = parser.parse_args()

