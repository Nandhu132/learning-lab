import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai_tools import SerperDevTool
load_dotenv()

llm = LLM(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    api_key=os.getenv("AWS_ACCESS_KEY_ID"),          
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION", "us-east-1"),
    temperature=0.7
)

search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

planner_agent = Agent(
role="Content Planner",
goal="Plan engaging and factually accurate content on {topic}",
backstory="You're working on planning a blog article "
        "about the topic: {topic}."
        "You collect information that helps the "
        "audience learn something "
        "and make informed decisions."
        "Your work is the basis for "
        "the Content Writer to write an article on this topic.",
verbose =True,
llm=llm,
tools=[search_tool],
allow_delegation=False,
)

writer_agent = Agent(
role="Content Writer",
goal="Write insightful and factually accurate "
"opinion piece about the topic: {topic}",
backstory="You're working on a writing "
        "a new opinion piece about the topic: {topic}. "
        "You base your writing on the work of "
        "the Content Planner, who provides an outline "
        "and relevant context about the topic. "
        "You follow the main objectives and "
        "direction of the outline,"
        "as provide by the Content Planner."
        "You also provide objective and impartial insights"
        "and back them up with information "
        "provide by the Content Planner. "
        "You acknowledge in your opinion piece"
        "when your statements are opinions "
        "as opposed to objective statements.",
verbose =True,
allow_delegation=False,
llm=llm
)

editor_agent = Agent(
role="Editor",
goal="Edit a given blog post to align with "
"the writing style of the organization.",
backstory="You are an editor who receives a blog post "
        "from the Content Writer. "
        "Your goal is to review the blog post "
        "to ensure that it follows journalistic best practices,"
        "provides balanced viewpoints "
        "when providing opinions or assertions, "
        "and also avoids major controversial topics "
        "or opinions when possible.",
verbose =True,
allow_delegation=False,
llm=llm
)

planner_task = Task(
description=
        "1. Prioritize the latest trends, key players,"
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources.",
expected_output="A comprehensive content plan document"
        "with an outline, audience analysis,"
        "SEO keywords, and resources.",
agent=planner_agent,
)

write_task = Task(
description=
        "1. Use the content plan to craft a compelling " 
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice.\n",
expected_output="A well-written blog post "
        "in markdown format, ready for publication,"
        "each section should have 2 or 3 paragraphs.",
agent=writer_agent,
)

edit_task = Task(
description=
        ("Proofread the given blog post for "
        "grammatical errors and "
        "alignment with the brand's voice."),
expected_output="A well-written blog post in markdown format, "
        "ready for publication, "
        "each section should have 2 or 3 paragraphs.",
agent=editor_agent
)

crew = Crew(
    agents=(planner_agent, writer_agent, editor_agent),
    tasks=(planner_task, write_task, edit_task), 
    verbose =True
)
result = crew.kickoff(inputs={"topic": "Artificial Intelligence in Automobile in 2025"})