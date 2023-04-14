from typing import List
import openai
from utils import CustomCharacter

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

    def generate_story_plot(self, title, custom_characters: List[CustomCharacter]) -> List[str]:
        """ Returns list of story prompts.
        """
        # single character customization for now
        character = custom_characters[0]
        custom_key = character.custom_name or character.orig_name
        prompt = (
            f"Tell me the story of {title} with {custom_key} as the main character '{character.orig_name}', " +
            f"without using any pronouns and in exactly {self.num_sentences} sentences. " +
            f"Show me a numbered list of exactly {self.num_sentences} sentences."
        )
        print("Prompt: \n", prompt)
        response = self.query_chatgpt(prompt)
        plot = [" ".join(x.split(" ")[1:]) for x in response.split("\n")]
        print("Response: \n", plot)
        print(f"number of sentences generated = {len(plot)}")
        return plot

    def query_chatgpt(self, content):
        self.messages.append({"role": "user", "content": content})
        print("Querying ChatGPT", self.messages)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        chat_response = completion.choices[0].message.content
        print(f"ChatGPT: {chat_response}")
        return chat_response


if __name__ == "__main__":
    title = "Little Red Riding Hood"
    custom_characters = [
        CustomCharacter(orig_name="wolf", custom_name="Aspen", custom_img_dir="sample_images/aspen")
    ]
    num_sentences = 10

    story_builder = StoryBuilder(num_sentences=10)
    plot = story_builder.generate_story_plot(title, custom_characters)
    print(plot)
