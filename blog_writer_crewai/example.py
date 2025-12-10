import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM

# Load .env
load_dotenv()

llm = LLM( 
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"), 
    temperature=0.7 
)

translator = Agent(
    role="Language Translator",
    goal="Translate to Tamil naturally and accurately.",
    backstory="Expert Tamil translator.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

def translate_to_tamil(text):

    task = Task(
        description=f"""
        Translate this text to Tamil:

        {text}

        Rules:
        - Preserve meaning
        - Use natural Tamil
        - Keep proper nouns intact
        """,
        expected_output="Tamil translation.",
        agent=translator
    )

    crew = Crew(
        agents=[translator],
        tasks=[task],
        verbose=True
    )

# Kick off the translation process
    result = crew.kickoff()

    # Extract the raw text from the result
    if hasattr(result, 'raw'):
        translation = result.raw
    else:
        translation = str(result)

    return translation


# Test translation
text_to_translate = """
Hi! How are you?
I am Nandhini. Good to see you!
"""

print("\nOriginal Text:")
print(text_to_translate)

print("\nTranslation:")
translation = translate_to_tamil(text_to_translate)
print(translation)

