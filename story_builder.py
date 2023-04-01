import os
import openai
openai.api_key = "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" #os.getenv("OPENAI_API_KEY")

class StoryBuilder(object):
    """
    Uses ChatGPT API to generate story plot.

    Input:
    - title

    Output:
    - sentences that describe plot of story
    """
    def __init__(self, config):
        config.num = 10
        messages = [
            {"role": "system", "content": "You're a storyteller. You can tell me the story using only %d sentences, and give a rich implication of this story in one sentence."%num}
        ]

        # while True:
        content = input("User: ")
        messages.append({"role": "user", "content": content})

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )     

        chat_response = completion.choices[0].message.content
        print(f'ChatGPT: {chat_response}')
        return chat_response
        # messages.append({"role": "assistant", "content": chat_response})

