# from urllib.parse import urlparse, parse_qs

# url = "https://www.youtube.com/watch?v=Aj66OpCEadI"
# parsed = urlparse(url)
# video_id = parse_qs(parsed.query).get("v", [None])[0]
# print(video_id)

import re

def extract_video_id(input_text):
    # If it's a raw video ID
    if re.fullmatch(r'[0-9A-Za-z_-]{11}', input_text):
        return input_text

    # Try to extract from standard YouTube URLs
    regex_patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",                 # Matches standard watch URLs and embedded
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",             # Short links
        r"(?:youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})"   # Shorts
    ]

    for pattern in regex_patterns:
        match = re.search(pattern, input_text)
        if match:
            return match.group(1)

    return None  # Return None if no match found

video_id = extract_video_id('https://www.youtube.com/watch?v=LPZh9BOjkQs&t=1s')
print(video_id)
