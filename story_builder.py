from typing import List
import openai

DEFAULT_OPENAI_API_KEY = "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"  #os.getenv("OPENAI_API_KEY")
DEFAULT_NUM_SENTENCES = 10


class StoryBuilder(object):
    """
    Uses ChatGPT API to generate story plot.

    Input:
    - title

    Output:
    - sentences that describe plot of story
    """
    def __init__(self, **config):
        self.messages = [{
            "role": "system",
            "content": (
                "You're a children's storyteller. ")
        }]
        self.num_sentences = config.get("num_sentences", DEFAULT_NUM_SENTENCES)
        openai.api_key = config.get("openai_api_key", DEFAULT_OPENAI_API_KEY)

    def generate_story_plot(self, title, customization) -> List[str]:
        """ Returns list of story prompts.
        """
        # single character customization for now
        characters = list(customization.items())[0]
        name, custom_key = characters
        prompt = (
            f"Tell me the story of {title} with {custom_key} as the main character '{name}', " +
            f"without using any pronouns and in exactly {self.num_sentences} sentences. " +
            f"Show me a list of {self.num_sentences} sentences."
        )
        response = self.query_chatgpt(prompt)
        return [" ".join(x.split(" ")[1:]) for x in response.split("\n")]

    def get_characters_for_story(self, n_characters):
        prompt = f"""
        """
        return

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
    character_customization = {
        "wolf": "Simon"
    }
    num_sentences = 10

    story_builder = StoryBuilder(num_sentences=10)
    plot = story_builder.generate_story_plot(title, character_customization)
    print(plot)
