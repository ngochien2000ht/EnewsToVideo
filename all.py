import shutil
import convert
from moviepy.editor import *
import make_audio
import make_video
from down_news import download_newspaper

def create_folder():
    path_temp = "temp"
    path_down = "temp/down"
    path_imgs_news = "temp/imgs_news"
    path_audios = "temp/audios"
    path_video_final = 'static'

    try:
        if not os.path.exists(path_temp):
            os.mkdir(path_temp)
        if not os.path.exists(path_down):
            os.mkdir(path_down)
        if not os.path.exists(path_imgs_news):
            os.mkdir(path_imgs_news)
        if not os.path.exists(path_audios):
            os.mkdir(path_audios)
        if not os.path.exists(path_video_final):
            os.mkdir(path_video_final)
    except:
        pass
    return path_audios,path_video_final,path_temp


def make(link,device, model, tokenizer):
    path_audios,path_video_final,path_temp = create_folder()
    url = link
    namevideo_first,topic_title = download_newspaper(url)
    namevideo = namevideo_first.replace(' ','')
    sentences = convert.down_imgs_from_keywords(topic_title,device, model, tokenizer)

    # create audio
    mystring = " ".join(sentences)
    make_audio.audio(mystring)
    make_audio.creat_audio(path_audios, sentences)

    # create video
    folder_img = "temp/down"
    make_video.video_audio_final(folder_img, sentences, path_audios, path_video_final, namevideo)

    return str(namevideo+".mp4"), namevideo_first
def del_foldertemp(path_temp):
    try:
        shutil.rmtree(path_temp)
    except:
        pass

# device, model, tokenizer = convert.run_model()
# link = "https://vnexpress.net/my-noi-nga-muon-doi-luong-thuc-lay-vu-khi-trieu-tien-4587653.html"
# make(link,device, model, tokenizer)