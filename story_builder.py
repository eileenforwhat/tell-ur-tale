from typing import List

import openai
from utils import CustomCharacter, StoryPage, parse_line_to_story_page

DEFAULT_OPENAI_API_KEY = "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"  #os.getenv("OPENAI_API_KEY")
DEFAULT_NUM_SENTENCES = 10


class StoryBuilder(object):
    """ Uses ChatGPT API to generate story plot.
    """
    def __init__(self, **config):
        self.messages = [{
            "role": "system",
            "content": (
                "You are a storyteller who tells children's stories. "
                "You don't use pronouns. You are good at counting numbers. "
                "You will focus on getting the number of sentences correct. "
                "Be descriptive about appearance of character and background. "
                "Answer the following question."
            )
        }]
        self.num_sentences = config.get("num_sentences", DEFAULT_NUM_SENTENCES)
        openai.api_key = config.get("openai_api_key", DEFAULT_OPENAI_API_KEY)

    def generate_story_plot(self, title, custom_characters: List[CustomCharacter]) -> List[StoryPage]:
        """ Returns list of story prompts.
        """
        # single character customization for now
        character = custom_characters[0]
        custom_key = character.custom_name or character.orig_name
        prompt = (
            f"Write a book of the story {title} with {custom_key} as the main character '{character.orig_name}' "
            f"with {self.num_sentences} pages, one sentence per page, and one illustration per page. "
            f"Provide a prompt for a speech-to-image system in square brackets to generate each illustration. "
            f"Show me a numbered list of exactly {self.num_sentences} sentences. "
            "As an example, for the story of Jack and the Beanstalk, you would generate: \n "
            "1. On a small farm, Jack lived with his mother in a small cottage [Generate an image of the cottage]. \n"
            "2. They were poor and had no food to eat, so his mother asked Jack to sell their cow [Generate an image of Jack leading the cow to a market]. \n"
            "3. On the way to the market, Jack met a strange old man who gave him magic beans in exchange for the cow [Generate an image of the old man and Jack exchanging the cow and magic beans]. \n"
            "4. Jack's mother was angry with him and threw the beans out of the window [Generate an image of Jack's mother throwing the beans out of the window]. \n"
            "5. The next day, Jack saw a huge beanstalk had grown from the magic beans and it reached up to the sky [Generate an image of Jack looking up at the beanstalk]. \n"
            "6. Jack climbed up the beanstalk and reached a land of giants [Generate an image of Jack meeting the giant]. \n"
            "7. The giant's wife helped Jack by giving him food and hiding him from her husband [Generate an image of the giant's wife hiding Jack]. \n"
            "8. Jack stole a golden harp and ran away, but the harp called out to the giant [Generate an image of Jack running away with the harp]. \n"
            "9. The giant chased Jack down the beanstalk and Jack cut down the beanstalk, causing the giant to fall and die [Generate an image of Jack cutting down the beanstalk and the giant falling]. \n"
            "10. Jack returned to his mother and they lived happily ever after with the golden harp [Generate an image of Jack and his mother with the golden harp]. \n"
        )
        # prompt = (
        #     f"Tell me the story of {title} with {custom_key} as the main character '{character.orig_name}', " +
        #     f"without using any pronouns and in exactly {self.num_sentences} sentences. " +
        #     f"Show me a numbered list of exactly {self.num_sentences} sentences."
        # )
        response = self.query_chatgpt(prompt)
        lines = response.split("\n")

        pages = [parse_line_to_story_page(line) for line in lines]
        print(pages)
        print(f"number of sentences generated = {len(pages)}")
        return pages

    def query_chatgpt(self, content):
        self.messages.append({"role": "user", "content": content})
        print("Querying ChatGPT", self.messages)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        chat_response = completion.choices[0].message.content
        print(f"ChatGPT: \n{chat_response}")
        return chat_response


if __name__ == "__main__":
    title = "Little Red Riding Hood"
    custom_characters = [
        CustomCharacter(
            orig_name="wolf", orig_object="wolf", custom_name="Aspen", custom_img_dir="sample_images/aspen"
        )
    ]

    story_builder = StoryBuilder(num_sentences=10)
    plot = story_builder.generate_story_plot(title, custom_characters)
    print(plot)
