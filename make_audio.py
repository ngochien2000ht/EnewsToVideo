from gtts import gTTS


def creat_audio(folderauio, sentences):
    m = []
    for i in range(len(sentences)):
        output = gTTS(sentences[i], lang="vi", slow=False)
        audioname = "/audio" + str(str(i) + ".mp3")
        output.save(folderauio + audioname)
        m.append(folderauio + audioname)
    return m


def audio(sentences):
    output = gTTS(sentences, lang="vi", slow=False)
    audiott = "temp/audiototal.mp3"
    output.save(audiott)
    return audiott