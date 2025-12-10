"""
"""

import difflib
import re

from csucsf.config import DEFAULT_TRANSCRIPTS_DIR


def read_transcript(participant_id: str, transcript_type: str, transcripts_dir=DEFAULT_TRANSCRIPTS_DIR, cleaning=True):
    """
    Reads and processes a transcript file for a specific participant and transcript type.
    
    This function locates a transcript file based on participant ID and transcript type,
    reads its content, and optionally performs character cleaning. The function expects
    exactly one matching transcript file for the given participant ID.
    
    Parameters:
        participant_id (str): Unique identifier for the participant whose transcript to read
        transcript_type (str): Type of transcript to read. Must be one of:
            - "manual": Edited SALT transcripts
            - "whisper": Raw Whisper transcripts
            - "whisperQC": Edited Whisper transcripts
        transcripts_dir (Path, optional): Directory containing transcript folders. 
            Defaults to DEFAULT_TRANSCRIPTS_DIR
        cleaning (bool, optional): Whether to remove special characters from content. 
            Defaults to True
    
    Returns:
        tuple: A tuple containing:
            - Path: The path to the transcript file that was read
            - str: The content of the transcript file as a single string with lines joined by spaces
    
    Raises:
        ValueError: If zero or multiple transcript files are found for the participant
        KeyError: If an invalid transcript_type is provided
    """
    # Define folder names for each transcript type
    transcript_folders = {
        "manual": "Edited_SALT_Transcripts",
        "whisper": "Raw_Whisper_Transcripts_09-23-2025",
        "whisperQC": "Edited_Whisper_Transcripts_09-23-2025",
    }

    # Construct path to the specific transcript folder and find all transcripts
    transcript_folder = transcripts_dir / transcript_folders[transcript_type]
    filename = list(transcript_folder.glob(f"{participant_id}_*"))

    # Verify exactly one transcript file exists for this participant
    filename_num = len(filename)
    if filename_num != 1:
        raise ValueError(f"Found several or no transcripts for {participant_id} in {transcript_folder}")

    # Read and process the transcript file
    with filename[0].open("r") as f:
        content = " ".join(line.strip() for line in f)

        if cleaning:
            # Characters to remove: symbols and BOM (Byte Order Mark)
            for char in ["=", "@", "&", "#", "%", "$", "\ufeff"]:
                content = content.replace(char, "")

    return filename[0], content


def show_comparison(s1, s2, output, width=40, margin=10):
    """
    function adapted from :
    https://medium.com/data-science/side-by-side-comparison-of-strings-in-python-b9491ac858
    """
    def tokenize(s):
        return re.split('\s+', s)
    
    def untokenize(ts):
        return ' '.join(ts)
    
    def equalize(s1, s2):
        l1 = tokenize(s1)
        l2 = tokenize(s2)
        res1 = []
        res2 = []
        prev = difflib.Match(0,0,0)
        for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
            if (prev.a + prev.size != match.a):
                for i in range(prev.a + prev.size, match.a):
                    res2 += ['_' * len(l1[i])]
                res1 += l1[prev.a + prev.size:match.a]
            if (prev.b + prev.size != match.b):
                for i in range(prev.b + prev.size, match.b):
                    res1 += ['_' * len(l2[i])]
                res2 += l2[prev.b + prev.size:match.b]
            res1 += l1[match.a:match.a+match.size]
            res2 += l2[match.b:match.b+match.size]
            prev = match
        return untokenize(res1), untokenize(res2)
    
    def insert_newlines(string, every=64, window=10):
        result = []
        from_string = string
        while len(from_string) > 0:
            cut_off = every
            if len(from_string) > every:
                while (from_string[cut_off-1] != ' ') and (cut_off > (every-window)):
                    cut_off -= 1
            else:
                cut_off = len(from_string)
            part = from_string[:cut_off]
            result += [part]
            from_string = from_string[cut_off:]
        return result
    
    s1, s2 = equalize(s1,s2)

    s1 = insert_newlines(s1, width, margin)
    s2 = insert_newlines(s2, width, margin)

    comparison = []
    for s1_line, s2_line in zip(s1, s2):
        left = s1_line.ljust(width)
        right = s2_line.ljust(width)
        comparison.append(f"{left} | {right} |")

    with output.open('w') as f:
        for line in comparison:
            f.write(f"{line}\n")


def create_html_from_text_files(file_paths, output_path):
    """
    Combines multiple text files into a single HTML page with file names as headings.
    
    Args:
        file_paths (list): List of paths to text files
        output_path (str): Path for output HTML file
    """
    # Start building HTML content
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Text Files</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        pre {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            white-space: pre-wrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Combined Text Files</h1>
"""
    
    # Process each file
    for file_path in file_paths:
        filename = file_path.name
        
        # Read file content
        with file_path.open('r') as file:
            content = file.read()
        
        # Add file section to HTML
        html_content += f"""
        <div class="file-section">
            <h2>{filename}</h2>
            <pre>{content}</pre>
        </div>
"""
    
    # Close HTML tags
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with output_path.open('w', encoding='utf-8') as html_file:
        html_file.write(html_content)
