import yt_dlp
from pydub import AudioSegment
import os

def download_youtube_playlist(playlist_url, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'ignoreerrors': True,  # Skip unavailable videos
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

# Example usage
playlist_url = "https://www.youtube.com/playlist?list=PLqTQYWmDZnP36NDszSwfRKw2f6qc5nOvd"
output_directory = r"C:\Users\lukas\Music\youtube_playlist"
download_youtube_playlist(playlist_url, output_directory)