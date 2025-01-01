import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import yt_dlp

def download_and_convert_playlist(playlist_url, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Updated options for yt-dlp with proper download settings
    ydl_opts = {
        'format': 'bestaudio/best',
        'paths': {'home': output_folder},
        'outtmpl': {
            'default': '%(title)s.%(ext)s',
            'temp': '%(title)s.%(ext)s.tmp'
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'writethumbnail': False,
        'ignoreerrors': True,
        'extract_flat': False,  # Changed from 'in_playlist' to False
        'quiet': False,
        'no_warnings': False,
        'verbose': True,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'skip_unavailable_fragments': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        'extractor_args': {
            'youtube': {
                'skip': ['dash', 'hls'],
                'player_skip': ['js', 'configs', 'webpage']
            }
        },
        'concurrent_fragment_downloads': 1
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Download playlist info first
            info = ydl.extract_info(playlist_url, download=False)
            if not info:
                print("Could not retrieve playlist information")
                return

            # Process each video in the playlist
            for entry in info['entries']:
                if entry is None:
                    continue
                
                try:
                    # Download single video
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    ydl.download([video_url])
                    print(f"Successfully downloaded: {entry.get('title', 'unknown')}")
                except Exception as e:
                    print(f"Error downloading {entry.get('title', 'unknown')}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing playlist: {e}")
            return
        
        # Process each downloaded video
        for entry in info.get('entries', []):
            # Skip if entry is None (happens for private/unlisted videos)
            if entry is None:
                print("Skipping unavailable video (might be private or unlisted)")
                continue
                
            # Skip if video is private or unlisted
            if entry.get('availability', '') in ['private', 'unlisted']:
                print(f"Skipping {entry.get('title', 'unknown')} - Video is {entry.get('availability')}")
                continue
                
            if not entry or 'filepath' not in entry:
                print(f"Skipping entry {entry.get('title', 'unknown')} (missing file path)")
                continue
            
            # Rest of the processing code
            input_file = entry['filepath']
            video_title = entry.get('title', 'unknown').replace("/", "_")
            output_file = os.path.join(output_folder, f"{video_title}.wav")
            
            if not os.path.exists(input_file):
                print(f"Input file does not exist: {input_file}")
                continue
            
            try:
                print(f"Converting {input_file} to {output_file}...")
                clip = VideoFileClip(input_file)
                clip.audio.write_audiofile(output_file)
                clip.close()
                print(f"Successfully converted: {output_file}")
            except Exception as e:
                print(f"Error converting {input_file} to WAV: {e}")

if __name__ == "__main__":
    # Use your provided playlist URL
    playlist_url = "https://www.youtube.com/watch?v=KzJYoPP5d8E&list=PLr_Nb1pGEmOSO38ZG6WSz1I0hE8f23oKr"
    # Output folder to save WAV files
    output_folder = os.path.expanduser("~/Music/taylor_swift_songs")
    
    download_and_convert_playlist(playlist_url, output_folder)
