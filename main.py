import whisper
import os

# Add directory into content folder
checkDownLoadFolder = os.path.exists("download")
if not checkDownLoadFolder:
  os.mkdir("download")

fileName = "recordings/GMT20220928-075224_Recording.m4a"

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(fileName)

outputTextsArr = []
while audio.size > 0:
  tirmedAudio = whisper.pad_or_trim(audio)
  # trimedArray.append(tirmedAudio)
  startIdx = tirmedAudio.size
  audio = audio[startIdx:]

  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(tirmedAudio).to(model.device)

  # detect the spoken language
  _, probs = model.detect_language(mel)
  # print(f"Detected language: {max(probs, key=probs.get)}")

  # decode the audio
  options = whisper.DecodingOptions()
  result = whisper.decode(model, mel, options)

  # print the recognized text
  outputTextsArr.append(result.text)

outputTexts = ' '.join(outputTextsArr)
print(outputTexts)

# Write into a text file
with open(f"download/{fileName}.txt", "w") as f:
  f.write(f"▼ Transcription of {fileName}\n")
  f.write(outputTexts)
