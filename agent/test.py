from textwrap import dedent
from typing import List

from agno.agent import Agent, RunResponse  # noqa
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field


class MovieScript(BaseModel):
    setting: str = Field(
        ...,
        description="A richly detailed, atmospheric description of the movie's primary location and time period. Include sensory details and mood.",
    )
    ending: str = Field(
        ...,
        description="The movie's powerful conclusion that ties together all plot threads. Should deliver emotional impact and satisfaction.",
    )
    genre: str = Field(
        ...,
        description="The film's primary and secondary genres (e.g., 'Sci-fi Thriller', 'Romantic Comedy'). Should align with setting and tone.",
    )
    name: str = Field(
        ...,
        description="An attention-grabbing, memorable title that captures the essence of the story and appeals to target audience.",
    )
    characters: List[str] = Field(
        ...,
        description="4-6 main characters with distinctive names and brief role descriptions (e.g., 'Sarah Chen - brilliant quantum physicist with a dark secret').",
    )
    storyline: str = Field(
        ...,
        description="A compelling three-sentence plot summary: Setup, Conflict, and Stakes. Hook readers with intrigue and emotion.",
    )


def json_mode_agent_test(location: str, debug: bool = True, tui: bool = True):
    """Agent that uses JSON mode for movie script generation"""
    json_mode_agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description=dedent("""\
            You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
            With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
            you craft unique stories that captivate audiences worldwide.

            Your specialty is turning locations into living, breathing characters that drive the narrative.\
        """),
        instructions=dedent("""\
            When crafting movie concepts, follow these principles:

            1. Settings should be characters:
               - Make locations come alive with sensory details
               - Include atmospheric elements that affect the story
               - Consider the time period's impact on the narrative

            2. Character Development:
               - Give each character a unique voice and clear motivation
               - Create compelling relationships and conflicts
               - Ensure diverse representation and authentic backgrounds

            3. Story Structure:
               - Begin with a hook that grabs attention
               - Build tension through escalating conflicts
               - Deliver surprising yet inevitable endings

            4. Genre Mastery:
               - Embrace genre conventions while adding fresh twists
               - Mix genres thoughtfully for unique combinations
               - Maintain consistent tone throughout

            Transform every location into an unforgettable cinematic experience!\
        """),
        response_model=MovieScript,
        use_json_mode=True,
        debug_mode=debug,
    )

    if tui:
        json_mode_agent.print_response(location, stream=True)
    else:
        response = json_mode_agent.run(location, stream=True)
        for chunk in response:
            print(chunk.content, end="", flush=True)


def structured_output_agent_test(location: str, debug: bool = True, tui: bool = True):
    """Agent that uses structured outputs for movie script generation"""
    structured_output_agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        description=dedent("""\
            You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
            With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
            you craft unique stories that captivate audiences worldwide.

            Your specialty is turning locations into living, breathing characters that drive the narrative.\
        """),
        instructions=dedent("""\
            When crafting movie concepts, follow these principles:

            1. Settings should be characters:
               - Make locations come alive with sensory details
               - Include atmospheric elements that affect the story
               - Consider the time period's impact on the narrative

            2. Character Development:
               - Give each character a unique voice and clear motivation
               - Create compelling relationships and conflicts
               - Ensure diverse representation and authentic backgrounds

            3. Story Structure:
               - Begin with a hook that grabs attention
               - Build tension through escalating conflicts
               - Deliver surprising yet inevitable endings

            4. Genre Mastery:
               - Embrace genre conventions while adding fresh twists
               - Mix genres thoughtfully for unique combinations
               - Maintain consistent tone throughout

            Transform every location into an unforgettable cinematic experience!\
        """),
        response_model=MovieScript,
        debug_mode=debug,
    )

    if tui:
        structured_output_agent.print_response(location, stream=True)
    else:
        response = structured_output_agent.run(location, stream=True)
        for chunk in response:
            print(chunk.content, end="", flush=True)


# Example usage with different locations
if __name__ == "__main__":
    json_mode_agent_test("Tokyo")
    structured_output_agent_test("Ancient Rome")

# More examples to try:
"""
Creative location prompts to explore:
1. "Underwater Research Station" - For a claustrophobic sci-fi thriller
2. "Victorian London" - For a gothic mystery
3. "Dubai 2050" - For a futuristic heist movie
4. "Antarctic Research Base" - For a survival horror story
5. "Caribbean Island" - For a tropical adventure romance
"""

# To get the response in a variable:
# from rich.pretty import pprint

# json_mode_response: RunResponse = json_mode_agent.run("New York")
# pprint(json_mode_response.content)
# structured_output_response: RunResponse = structured_output_agent.run("New York")
# pprint(structured_output_response.content)
