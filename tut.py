from story_builder import StoryBuilder


def create_illustrator(config):
    # instantiate type of illustrator from yaml config
    pass


def create_story_builder(config):
    pass


class TellUrTalePipeline(object):
    def __init__(self, config):
        self.config = config
        self.story_builder = create_story_builder(config.story_builder)
        self.illustrator = create_illustrator(config.ilustrator)

    def run_tut(self):
        # build story
        # call illustrator
        pass


if __name__ == '__main__':
    pipeline = TellUrTalePipeline(...)
    pipeline.run_tut()
