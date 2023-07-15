import shutil
import convert
import tkinter as tk
from tkinter import filedialog
from moviepy.editor import *
import make_audio
import make_video
from down_news import download_newspaper


def OnButtonClick():
    global folder_save
    folder_save = filedialog.askdirectory()


def make():
    url = link.get()
    namevideo,topic_title = download_newspaper(url)
    sentences = convert.down_imgs_from_keywords(topic_title)

    # create audio
    mystring = " ".join(sentences)
    make_audio.audio(mystring)
    make_audio.creat_audio(path_audios, sentences)

    # create video
    folder_img = "temp/down"
    make_video.video_audio_final(folder_img, sentences, path_audios, folder_save, namevideo)

    # shutil.rmtree(path_temp)
    quit()


path_temp = "temp"
path_down = "temp/down"
path_imgs_news = "temp/imgs_news"
path_audios = "temp/audios"

try:
    if not os.path.exists(path_temp):
        os.mkdir(path_temp)
    if not os.path.exists(path_down):
        os.mkdir(path_down)
    if not os.path.exists(path_imgs_news):
        os.mkdir(path_imgs_news)
    if not os.path.exists(path_audios):
        os.mkdir(path_audios)
except:
    pass


def quit():
    wd.quit()


wd = tk.Tk()
wd.title('news_to_video')
wd.geometry("300x120")
wd.wm_maxsize(width=300, height=120)
tk.Button(wd, width=30, text='Chosse Directory to save',
          command=OnButtonClick).grid(row=1, column=1)
tk.Label(wd, width=30, text="        Link of vnExpress newspaper:", anchor='w').grid(row=2, column=1)
link = tk.StringVar()
tk.Entry(wd, width=35, textvariable=link).grid(row=3, column=1)
tk.Button(wd, width=10, text="OK", command=make).grid(row=6, column=2)
tk.Button(wd, width=10, text='Quit', command=wd.destroy).grid(row=7, column=2)

wd.mainloop()