import os
import shutil
import time
from flask import Flask, render_template, request
import all
import convert
from functools import lru_cache
app = Flask(__name__)

@lru_cache()
def load_model():
    device, model, tokenizer = convert.run_model()
    return device, model, tokenizer



@app.route('/', methods=['GET', 'POST'])
def index():

    device, model, tokenizer = load_model()
    if request.method == 'POST':
        temp_folder = 'temp'
        # a = time.time()
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.mkdir(temp_folder)
        link = request.form['video_name']
        video_name, topic = all.make(link, device, model, tokenizer)
        # b = time.time()
        # print(b-a)
        return render_template('index.html', video_name=video_name, topic=topic)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
