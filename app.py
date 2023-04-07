import os
import yaml
from flask import Flask, render_template
from tut import TellUrTalePipeline
from time import time

app = Flask(__name__)

@app.route("/")
def generate_images():
    # make a new folder
    output_dir = f"static/output/tmp{int(time())}"
    pass

    # run pipeline that saves images into created folder
    with open('config.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
    config['output_dir'] = output_dir
    print(f"Running {config['project_name']}...")
    
    pipeline = TellUrTalePipeline(config)
    pipeline.run_tut()

    # return images from that pipeline
    image_filepaths = sorted(os.listdir(output_dir))
    image_fullpaths = list(map(lambda file_name: os.path.join(output_dir, file_name), image_filepaths))
    return render_template('story.html', images=image_fullpaths)



