import openai

# Ensure your API key is properly set in your environment
import os
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def generate_story(caption):
    prompt = """
    You are a story teller; 
    You can generate a short story based on a simple narrative, the story should be no more than 100 words with simple vocabulary;
    CONTEXT: {scenario}
    STORY:
    """.format(scenario=caption)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

# Example usage
caption = "a photography of a man holding a child in his arms. There is a man holding a small child in his arms."
story = generate_story(caption)
print(story)
