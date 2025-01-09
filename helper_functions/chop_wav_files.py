import os
import wave
import contextlib

def chop_wav_file(file_path, output_dir, segment_length=30):
    with wave.open(file_path, 'rb') as audio:
        frame_rate = audio.getframerate()
        num_frames = audio.getnframes()
        duration = num_frames / frame_rate
        segment_length_frames = segment_length * frame_rate
        num_segments = int(duration // segment_length)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i in range(num_segments):
            start_frame = i * segment_length_frames
            end_frame = start_frame + segment_length_frames
            audio.setpos(int(start_frame))
            segment_frames = audio.readframes(int(segment_length_frames))
            
            segment_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_segment_{i+1}.wav"
            segment_path = os.path.join(output_dir, segment_name)
            
            with wave.open(segment_path, 'wb') as segment:
                segment.setnchannels(audio.getnchannels())
                segment.setsampwidth(audio.getsampwidth())
                segment.setframerate(frame_rate)
                segment.writeframes(segment_frames)
            
            print(f"Exported {segment_name}")

def chop_wav_files_in_folder(input_folder, output_folder, segment_length=30):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            chop_wav_file(file_path, output_folder, segment_length)

# Example usage
input_folder = r"C:\Users\lukas\Music\youtube_playlist"
output_folder = r"C:\Users\lukas\Music\youtube_playlist_chopped"
chop_wav_files_in_folder(input_folder, output_folder)