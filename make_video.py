import shutil
from PIL import Image
from moviepy.editor import *
import random


def copyimg(folder_video, target):
    for filename in os.listdir(folder_video):
        shutil.copyfile(folder_video + filename, target + filename)


def size_all_imgs(f):
    for folder in os.listdir(f):
        for file in os.listdir(f + "/" + folder):
            f_img = f + "/" + folder + "/" + file
            try:
                img = Image.open(f_img)
                img = img.resize((1200, 800))
                img.save(f_img)
            except:
                os.remove(f_img)


def size_imgs(f):
    for file in os.listdir(f):
        f_img = f + "/" + file
        try:
            img = Image.open(f_img)
            img = img.resize((1200, 800))
            img.save(f_img)
        except:
            os.remove(f_img)


def creat_video(sentences, nameaudio):
    clips = []
    for i in range(len(sentences)):
        try:
            folder_video = "temp/down/" + str(i) + "/"
            filevideo = os.listdir(folder_video)[0]
            img = ImageClip(folder_video + filevideo)
            audio = AudioFileClip(nameaudio + "/" + os.listdir(nameaudio)[i])
            clip = img.set_duration(
                audio.duration)
            audio.close()
            img.close()
        except:
            folder_video = "temp/imgs_news/"
            filevideo = os.listdir(folder_video)[random.choice(range(len(os.listdir(folder_video))))]
            img = ImageClip(folder_video + filevideo)
            audio = AudioFileClip(nameaudio + "/" + os.listdir(nameaudio)[i])
            clip = img.set_duration(
                audio.duration)
            audio.close()
            img.close()
        clips.append(clip)
    return clips


def creat_video_final(clips):
    video_clip = concatenate_videoclips(clips)
    for clip in clips:
        clip.close()
    video_clip.write_videofile("temp/video-output.mp4", fps=2, remove_temp=True, codec="libx264", audio_codec="aac")


def video_audio_final(folder_img, sentences, nameaudio, pathsave, namevideo):
    size_all_imgs(folder_img)
    size_imgs("temp/imgs_news")
    clips = creat_video(sentences, nameaudio)
    creat_video_final(clips)
    videoclip = VideoFileClip("temp/video-output.mp4")
    audioclip = AudioFileClip("temp/audiototal.mp3")
    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip = videoclip.fx(vfx.speedx, 1.40)
    videoclip.write_videofile(pathsave + "/uploads/" + str(namevideo) + ".mp4")
    audioclip.close()
    videoclip.close()