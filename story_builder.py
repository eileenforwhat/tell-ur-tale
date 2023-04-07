import openai
openai.api_key = "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"  #os.getenv("OPENAI_API_KEY")


class StoryBuilder(object):
    """
    Uses ChatGPT API to generate story plot.

    Input:
    - title

    Output:
    - sentences that describe plot of story
    """
    def __init__(self, **kwargs):
        self.messages = [{
            "role": "system",
            "content": (
                "You're a children's storyteller. ")
        }]

    def generate_story_plot(self, title, character_custom_key, character_name, n_sentences):
        """
        Returns list of sentences

        :param title:
        :param key:
        :param character_name:
        :param n_sentences:
        :return:
        """
        prompt = (
            f"Tell me the story of {title} with {character_custom_key} as the main character '{character_name}', " +
            f"without using any pronouns and in exactly {n_sentences} sentences. " +
            f"Show me a list of {n_sentences} sentences."
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
    character_custom_key = "Simon"
    character_name = "wolf"
    num_sentences = 10

    story_builder = StoryBuilder()
    plot = story_builder.generate_story_plot(
        title, character_custom_key, character_name, num_sentences
    )
    print(plot)
