from pydantic import BaseModel
import os
import re
import json
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Ensure tools.py is in the same directory

load_dotenv()

# Define response model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Set up the LLM
llm = ChatCohere(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    temperature=0,
    model="command-r"
)

# Parser for the response (we might not use this directly in the function anymore)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Updated prompt to encourage using all tools and format output correctly
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful research assistant.
Use all available tools (`search`, `wikipedia`, `save_text_to_file`) to gather very detailed and very long information before responding.
When you provide your summary, **always include the specific URLs of the most relevant web pages or citations that support each piece of information in your summary within the 'sources' list.** Aim for accuracy and verifiability.
Respond in **JSON format only**, with these keys:
- "topic": Main research topic
- "summary": Concise summary of findings
- "sources": List of the specific URLs or citations used to support the summary
- "tools_used": List of tools used (e.g., "search", "wikipedia", "save_text_to_file")"""
         ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Register tools (use the imported ones)
tools = [search_tool, wiki_tool, save_tool]

# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def research_and_extract(query: str) -> dict:
    """
    Performs research with aggressive cleaning, targeting leading whitespace.
    """
    try:
        raw_response = agent_executor.invoke({"query": query})
        output_text = raw_response["output"].strip()

        # Remove any leading/trailing code block markers
        cleaned_output = re.sub(r"^```(?:json)?\s*", "", output_text, flags=re.IGNORECASE).strip()
        cleaned_output = re.sub(r"\s*```$", "", cleaned_output, flags=re.IGNORECASE).strip()

        

        # Attempt to find the first and last curly braces to isolate the JSON
        start_index = cleaned_output.find('{')
        end_index = cleaned_output.rfind('}')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_string = cleaned_output[start_index : end_index + 1]
            

            # Remove leading whitespace from each line
            json_string_cleaned_lines = "\n".join(line.lstrip() for line in json_string.splitlines())
            #print("\nExtracted JSON String (After Line Cleaning):\n", json_string_cleaned_lines)

            #print("\nExtracted JSON String (Raw Keys):\n", {repr(k): v for k, v in json.loads(json_string_cleaned_lines).items()})

            structured_data = json.loads(json_string_cleaned_lines)
            return {
                "summary": structured_data.get("summary", ""),
                "sources": structured_data.get("sources", []),
                "tools_used": structured_data.get("tools_used", []),
            }
        else:
            print("⚠️ Could not reliably isolate JSON object in the output.")
            return None

    except json.JSONDecodeError as json_e:
        print(f"❌ JSONDecodeError: {json_e}")
        print("⚠️ Problematic JSON String:\n", json_string if 'json_string' in locals() else "No JSON string found")
        print("🔎 Original Raw Output:\n", raw_response.get("output"))
        return None
    except Exception as e:
        print(f"❌ Error processing response: {e}")
        print("🔎 Original Raw Output:\n", raw_response.get("output"))
        return None
if __name__ == "__main__":
    research_query = input("What can I help you research? ")
    research_result = research_and_extract(research_query)

    if research_result:
        print("\n--- Research Results ---")
        print("Summary:", research_result["summary"])
        print("Sources:", research_result["sources"])
        print("Tools Used:", research_result["tools_used"])

        
        