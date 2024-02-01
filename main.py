import os
import pydub
import shutil
from tqdm import tqdm
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_silence
from googletrans import Translator
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import pipeline
import torchaudio
import speech_recognition as sr

### Give your video filepath hear
video_filepath = r"C:\Users\dheer\OneDrive\Documents\Resume_Project\Video_dub\input_output\PM Modi addresses INTERPOL General Assembly in New Delhi.mp4"

### extract video name for output
video_name = os.path.basename(video_filepath)

base_directory = os.path.dirname(video_filepath)

### provide the necessary links to run model
audio_folder_imp = r"C:\Users\dheer\OneDrive\Documents\Resume_Project\Video_dub\audio_processing\audio_file"
audio_chunks_input = r"C:\Users\dheer\OneDrive\Documents\Resume_Project\Video_dub\audio_processing\audio_chunks_input"
audio_chunks_output = r"C:\Users\dheer\OneDrive\Documents\Resume_Project\Video_dub\audio_processing\audio_chunks_output"
temp_file_path = os.path.join(audio_folder_imp ,"temp_tts.mp3")
output_audio = os.path.join(audio_folder_imp ,"smooth_merged_audio.wav")
extracted_audio = os.path.join(audio_folder_imp ,"extracted_audio.wav")
target_audio_path = extracted_audio

def remove_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

remove_files(audio_chunks_input)
remove_files(audio_chunks_output)

## Input language selection
input_lang = -1
print("\nInput Language: \n 1. English \n 2. Hindi \n 3. Marathi")
while input_lang not in (1, 2, 3):
    user_input = input("\nEnter Input Language Choice: ")
    if user_input.isdigit():
        input_lang = int(user_input)
        if input_lang not in (1, 2, 3):
            print("Invalid choice. Please enter 1 for English or 2 for Hindi or 3 for Marathi.")
        else:
            if input_lang == 1:
                src = 'en'
                print("\nInput Language : English")
            elif input_lang == 2:
                src = 'hi'
                print("\nInput Language : Hindi")
            elif input_lang == 3:
                src = 'mr'
                print("\nInput Language : Marathi")
    else:
        print("Invalid input. Please enter a valid numeric choice (1 or 2 or 3).")
            
def other_languages(output):

    if output == 3:
        dest = "gu" 
        output_lang = "gu"
        print("\nOuptut Language : Gujarati")
        return dest, output_lang

    elif output == 4:
        dest = "bn" 
        output_lang = "bn"
        print("\nOuptut Language : Bengali")
        return dest, output_lang

    elif output == 5:
        dest = "ml"
        output_lang = "ml"
        print("\nOuptut Language : Malayalam")
        return dest, output_lang

    elif output == 6:
        dest = "ta" 
        output_lang = "ta"
        print("\nOuptut Language : Tamil")
        return dest, output_lang

    elif output == 7:
        dest = "te" 
        output_lang = "te"
        print("\nOutput Language : Telugu")
        return dest, output_lang

    elif output == 8:
        dest = "kn" 
        output_lang = "kn"
        print("\nOuptut Language : Kannada")
        return dest, output_lang


output = -1
if input_lang == 1:
    print("\nOutput Language:\n 1. Hindi\n 2. Marathi\n 3. Gujarati\n 4. Bengali\n 5. Malayalam\n 6. Tamil\n 7. Telugu\n 8. Kannada")
    
    while output not in (1,2,3,4,5,6,7,8):
        user_input = input("\nEnter your choice: ")
        if user_input.isdigit():
            output = int(user_input)
            if output not in (1,2,3,4,5,6,7,8):
                print("Invalid input. Please enter a valid numeric choice between (1 to 8).")
            else:
                if output == 1:
                    dest = "hi" 
                    output_lang = "hi"
                    print("\nOuptut Language : Hindi")
                elif output == 2:
                    dest = "mr" 
                    output_lang = "mr"
                    print("\nOuptut Language : Marathi")

                else:
                    dest, output_lang = other_languages(output)
        else:
            print("Invalid input. Please enter a valid numeric choice between (1 to 8).")
         
elif input_lang == 2:
    print("\nOutput Language:\n 1. English\n 2. Marathi\n 3. Gujarati\n 4. Bengali\n 5. Malayalam\n 6. Tamil\n 7. Telugu\n 8. Kannada")
    
    while output not in (1,2,3,4,5,6,7,8):
        user_input = input("\nEnter your choice: ")
        if user_input.isdigit():
            output = int(user_input)
            if output not in (1,2,3,4,5,6,7,8):
                print("Invalid input. Please enter a valid numeric choice between (1 to 8).")
            else:
                if output == 1:
                    dest = "en"
                    output_lang = "en"
                    print("\nOuptut Language : English")

                elif output == 2:
                    dest =  "mr"
                    output_lang = "mr"
                    print("\nOuptut Language : Marathi")

                else:
                    dest, output_lang = other_languages(output)
        else:
            print("Invalid input. Please enter a valid numeric choice between (1 to 8).")
else:
    print("\nOutput Language:\n 1. Hindi\n 2. English\n 3. Gujarati\n 4. Bengali\n 5. Malayalam\n 6. Tamil\n 7. Telugu\n 8. Kannada")
    
    while output not in (1,2,3,4,5,6,7,8):
        user_input = input("\nEnter your choice: ")
        if user_input.isdigit():
            output = int(user_input)
            if output not in (1,2,3,4,5,6,7,8):
                print("Invalid input. Please enter a valid numeric choice between (1 to 8).")
            else:
                if output == 1:
                    dest = "hi" 
                    output_lang = "hi"
                    print("\nOuptut Language : Hindi")

                elif output == 2:
                    dest = "en"
                    output_lang = "en"
                    print("\nOuptut Language : English")

                else:
                    dest, output_lang = other_languages(output)
        else:
            print("Invalid input. Please enter a valid numeric choice between (1 to 8).") 
"""
Step 1: Take audio file from the video
        and store the audio for further processing

"""
# Load the video file
video_file = video_filepath
video_clip =  VideoFileClip(video_file)

# Get the audio from the video clip
audio_clip = video_clip.audio

# Export the audio to an audio file (e.g., MP3 format)
output_audio_file = extracted_audio
audio_clip.write_audiofile(output_audio_file)

# Adding silence to manage audio last at the end
silence= AudioSegment.silent(duration=2000)
audio_weights = AudioSegment.from_file(output_audio_file)
final_audio = audio_weights + silence
final_audio.export(output_audio_file, format="wav")

"""

Step 2: From stored audio 
1. Make audio chunks on the silence (Max chunk size 30 sec)
2. Load the chunk and then do STT (HuBERT model from Transformers)
3. Translate the English Text to Hindi (translate library)
4. Perform TTS and store the audio chunk

"""


### END OF TTS AND VOICE CLONING ### 
def stt_google(audio_file_path, lang):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio, language=lang + '-IN')
        return transcription
    except sr.UnknownValueError:
        # Handle the unknown value error here (e.g., return None)
        return None
    except sr.RequestError as e:
        # Handle other potential errors (e.g., network issues)
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


### SPEEDUP AUDIO if(output audio is less than target duration) ###
def speedup_audio (output_filepath, output_duration):
    input_duration = get_audio_duration(output_filepath)

    if(input_duration-output_duration == 1):
        playback_speed = (output_duration-1) / input_duration
        playback_speed = 2.0 - playback_speed
    else:
       playback_speed = input_duration/output_duration
    # playback_speed = input_duration/output_duration
    
    audio = AudioSegment.from_file(output_filepath)
    sped_up_audio = audio.speedup(playback_speed)
    sped_up_audio.export(output_filepath, format="wav")

### END OUT SPEEDUP AUDIO###
def google_tts(input_text, output_language,output_audio_path): 
    tts = gTTS(text=input_text, lang=output_language)
    # Save the speech as an audio file
    tts.save(output_audio_path)

def generate_tts_audio(wav_file_path, wav_output_path, text2, output_lang, target_path):

    target_duration = get_audio_duration(wav_file_path)
    target_duration_ms = get_audio_duration_ms(wav_file_path)

    google_tts(text2, output_lang, wav_output_path)
    # voice_cloning(wav_output_path,target_path)
    output_audio_duration = get_audio_duration(wav_output_path)
    
    audio = AudioSegment.from_file(wav_output_path)
    
    if output_audio_duration == target_duration:
        return
    elif output_audio_duration < target_duration:
        # padding = AudioSegment.silent(duration=target_duration - len(audio))
        padding = AudioSegment.silent(duration=(target_duration_ms - len(audio)) / 2)
    
        final_audio = padding + audio + padding

        # Save the generated audio as a WAV file
        final_audio.export(wav_output_path, format='wav')
        return
    elif output_audio_duration > target_duration:
    
        speedup_audio(wav_output_path, target_duration)
        return
    
def translate_to_dest(text, s, d):
    translator = Translator()
    translation = translator.translate(text, src=s, dest=d)
    return translation.text
    

def get_audio_duration(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)/1000
    return duration

def get_audio_duration_ms(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    return duration

def stt_whisper(wav_file_path):
    try:
        # Load the WAV file using torchaudio and downmix it to a single channel
        input_waveform, sample_rate = torchaudio.load(wav_file_path)
        input_waveform = input_waveform.mean(dim=0)  # Downmix to a single channel
        # Transcribe the entire audio without chunking
        prediction = pipe(input_waveform.numpy(), return_timestamps=True)["text"]
        return prediction
    except Exception as e:
        return None

def adjust_sample_rate(input_audio_path, output_audio_path, target_sample_rate):
    # Load the input audio file using pydub
    input_audio = AudioSegment.from_file(input_audio_path)
    # Resample the audio to the target sample rate
    output_audio = input_audio.set_frame_rate(target_sample_rate)
    # Export the adjusted audio to the output file
    output_audio.export(output_audio_path, format='wav')

def export_chunk(audio_file,output_file, audio_name, start, end):
    large_audio = AudioSegment.from_wav(audio_file)

    silence_duration = 500  # in milliseconds
    silence_start = AudioSegment.silent(duration=silence_duration)
    silence_end = AudioSegment.silent(duration=silence_duration)

    start_time = start  # Start time in milliseconds
    end_time = end    # End time in milliseconds

    # Extract the desired chunk of audio
    desired_chunk = large_audio[start_time:end_time]

    desired_chunk = desired_chunk.set_frame_rate(16000)

    final_audio = silence_start + desired_chunk + silence_end

    output_path = os.path.join(output_file, audio_name)
    final_audio.export(output_path, format="wav")

### MERGING ALL AUDIO CHUNKS USING OVERLAY ###
def overlay_audio(input_audio1, input_audio2, output_path):
    # Load audio1 and audio2 (replace with your file paths)
    audio1 = AudioSegment.from_file(input_audio1)
    audio2 = AudioSegment.from_file(input_audio2)

    # Ensure both audio segments have the same sample rate (if not, you can use .set_frame_rate())
    if audio1.frame_rate != audio2.frame_rate:
        audio2 = audio2.set_frame_rate(audio1.frame_rate)

    temp_len = len(audio1)-1000
    audio1 = audio1 + AudioSegment.silent(len(audio2)-1000)

    audio2 = AudioSegment.silent(temp_len) + audio2

    # Overlay audio2 on top of audio1
    output_audio = audio1.overlay(audio2)

    # Export the combined audio
    output_audio.export(output_path, format="wav")
def merge_audio(audio_chunk_list, output_path, chunk_path):

    chunk1 = os.path.join(chunk_path,audio_chunk_list[0])
    chunk2 = os.path.join(chunk_path,audio_chunk_list[1])
    overlay_audio(chunk1, chunk2, output_path)


    for i in range(2,len(audio_chunk_list)):
        chunk1 = output_path
        chunk2 = os.path.join(chunk_path,audio_chunk_list[i])
        overlay_audio(chunk1, chunk2, output_path)

### END OF MERGING CODE ###

def model_predict(wav_file_path,wav_output_path, output_lang, src, dest):

    if src == 'en':
        text = stt_whisper(wav_file_path)
        # text = stt_google(wav_file_path,"en")
    elif src == 'hi':
        # text = stt_azure(wav_file_path,"hi-IN")
        text = stt_google(wav_file_path,"hi")
    else:
        text = stt_google(wav_file_path,"mr")

    if(text == None):
        shutil.copyfile(wav_file_path, wav_output_path)
        return

    # Open the text file in write mode (creates the file if it doesn't exist)
    input_text_path = os.path.join(base_directory,(src+"_to_"+dest+"_input.txt"))
    # Open the file in append mode to add new content on a new line
    with open(input_text_path, "a", encoding="utf-8") as file:
        file.write(text + "\n")  # Add the new content and a newline character
    
    # Translate from source language to destination
    text2 = translate_to_dest(text, src, dest)

    output_text_path = os.path.join(base_directory,(src+"_to_"+dest+"_output.txt"))
    # Open the file in append mode to add new content on a new line
    with open(output_text_path, "a", encoding="utf-8") as file:
        file.write(text2 + "\n")  # Add the new content and a newline character

    generate_tts_audio(wav_file_path, wav_output_path, text2, output_lang, target_audio_path)

    return text2

# Load the audio file
audio_file = extracted_audio
# Export the desired chunk as a new audio file

audio = AudioSegment.from_wav(audio_file)

# Parameters for silence detection
window_size = 1000  #1 second in milliseconds
silence_threshold = -40  # Silence threshold in dBFS (adjust as needed)

# Iterate over the audio in steps of 1 second
silence = []
nonSilence = []
audio_chunk_list = []
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
)
for start in tqdm(range(0, len(audio), window_size)):
    end = start + window_size
    audio_window = audio[start:end]
    
    # Detect silence in the window
    silence_intervals = detect_silence(audio_window, min_silence_len=1000, silence_thresh=silence_threshold)

    if silence_intervals:
        if nonSilence:
            first = nonSilence[0][0]
            last = nonSilence[-1][-1]
            name = f"nonSilence_{first//1000}_{last//1000}.wav"
            export_chunk(audio_file, audio_chunks_input, name, first, last)
            model_wav_input = os.path.join(audio_chunks_input, name)
            model_wav_output = os.path.join(audio_chunks_output, name)
            model_predict(model_wav_input,model_wav_output, output_lang, src, dest)
            audio_chunk_list.append(name)
            nonSilence.clear()
        
        silence.append((start, end))
    else:
        if len(nonSilence) == 10:
            first = nonSilence[0][0]
            last = nonSilence[-1][-1]
            name = f"nonSilence_{first//1000}_{last//1000}.wav"
            export_chunk(audio_file, audio_chunks_input, name, first, last)
            model_wav_input = os.path.join(audio_chunks_input, name)
            model_wav_output = os.path.join(audio_chunks_output, name)
            model_predict(model_wav_input,model_wav_output, output_lang, src, dest)
            audio_chunk_list.append(name)
            nonSilence.clear()
        if silence:
            first = silence[0][0]
            last =  silence[-1][-1]
            name = f"silence_{first//1000}_{last//1000}.wav"
            export_chunk(audio_file, audio_chunks_input, name, first, last)
            export_chunk(audio_file, audio_chunks_output, name, first, last)
            audio_chunk_list.append(name)
            silence.clear()

        nonSilence.append((start, end))

"""

Step 3: Merge all the audio chunks of Hindi
        and Export the converted audio file.

"""

merge_audio(audio_chunk_list, output_audio, audio_chunks_output)

"""
Step 4: Add new Hindi audio to video file and export the audio

"""

# Load the video file
video_clip = VideoFileClip(video_file)

# Load the new audio file
new_audio_file = output_audio
new_audio_clip = AudioFileClip(new_audio_file)

# Remove the existing audio from the video
video_without_audio = video_clip.set_audio(None)

# Combine the video without audio with the new audio
final_video = video_without_audio.set_audio(new_audio_clip)
output_video = os.path.join(base_directory,(src+"_to_"+dest+"_"+video_name))
# Export the final video with the new audio
output_video_file = output_video
final_video.write_videofile(output_video_file, codec="libx264")