import requests
from bs4 import BeautifulSoup
from PIL import Image


def cut(coverpage_news, author):
    text = []
    cnews = []
    cnww = []
    for i in range(0, len(coverpage_news)):
        k1 = coverpage_news[i].get_text().replace("\n", "")
        cnews.append(k1)
    for i in range(0, len(author)):
        k2 = author[i].get_text().replace("\n", "")
        cnww.append(k2)
    for t in cnews:
        if t not in cnww:
            text.append(t)
    return text


def list_to_str(text):
    mystring = " ".join(text)
    mystring = mystring.replace(". ", ". \n")
    with open('temp/readme.txt', 'w+', encoding="utf8") as f:
        f.write(mystring)


def getlink_imgs_from_news(url):
    r1 = requests.get(url)
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, "lxml")
    imgs = soup1.find_all('meta', {'itemprop': 'url'})
    links = soup1.find_all('meta', {'itemprop': 'url', 'property': "og:url"})
    list_imgs = []
    list_links = []
    fn_imgs = []
    for img in imgs:
        if 'content' in img.attrs:
            img_ = img.get('content')
            list_imgs.append(img_)
    for link in links:
        if 'content' in link.attrs:
            link_ = link.get('content')
            list_links.append(link_)
    for i in list_imgs:
        if i not in list_links:
            fn_imgs.append(i)
    cmt_videos = soup1.find_all('div', {'class': 'inner_caption'})
    cmt_videos_imgs = soup1.find_all('p', {'class': 'Image'})
    cmt_imgs = cut(cmt_videos_imgs, cmt_videos)
    return fn_imgs, cmt_imgs


def down_imgs_from_news(fn_imgs):
    for i in range(len(fn_imgs)):
        img_url = fn_imgs[i]
        img = Image.open(requests.get(img_url, stream=True).raw)
        img.resize((3000, 2000))
        path = "temp/imgs_news"
        try:
            img.save(path + "/"+ str(i) + "img"  + ".jpg")
        except:
            img.save(path + "/"+ str(i) + "img"  + ".png")


def download_newspaper(url):
    r1 = requests.get(url)
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, "lxml")
    coverpage_news = soup1.find_all("p", {"class": "Normal"})
    author = soup1.find_all("p", {"class": "Normal", "style": "text-align:right;"})
    title_news = soup1.find_all("h1", {"class": "title-detail"})
    title = title_news[0].get_text()
    topic = soup1.find_all("meta",{"name":"its_subsection"})
    "content" in topic[0].attrs
    tt1 = topic[0].get('content')
    tp = tt1.split(", ")
    text = cut(coverpage_news, author)
    list_to_str(text)
    fn_imgs, cmt_imgs = getlink_imgs_from_news(url)
    down_imgs_from_news(fn_imgs)
    return title,tp[1]