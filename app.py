import os
from pathlib import Path
from openai import OpenAI
from llama_index import SimpleDirectoryReader
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for allowing all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including POST
    allow_headers=["*"],  # Allows all headers
)

topics_prompt = """
transcript:
{transcript}
=============

Act as a topic clustering expert proficient with identifying topic clusters that are sequential in time; span the complete set of relevant topics discussed; and maximize topic diversity.

Each topic should reference a discussion cluster that is germane to the podcast narrative and not non sequiturs or speaker actions or feelings. 

Produce an optimal topic list in json format with 2 key:value pairs per item:
1. time_range: The time range when the topic was discussed
2. topic_description: A detailed one-sentence topic description in the style of a heading for podcast notes.

There should be a minimum of 10 and a maximum of 15 topics. Produce your answer as valid json.
"""

summary_prompt = """
full transcript:
{text}
================

Act as a podcast section synthesizer. Given the above full transcript, please summarize the podcast discussion that occurs from {time_range} on the topic of {topic_statement}.  Always exhaustively and completely extract the most important and relevant parts and properly attribute speakers. Focus on key concepts, significant dialogues and debates, assertions, and any important details that are crucial to understanding the narrative and progression of the conversation. Remove redundant or irrelevant details to create a concise yet comprehensive summary that retains the core essence, facts, concepts, anecdotes and salient details shared by the guests in the target section.
"""

async def get_summary(client, text, time_range, topic_description, summary_prompt):
    print(f'Getting summary for topic: {topic_description}\n\n')
    summary_query = summary_prompt.format(text=text,time_range=time_range,topic_statement=topic_description)
    messages = [
        {"role": "user", "content": summary_query},
    ]

    completion = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4-1106-preview",
        temperature=0,
        messages=messages
    )
    response = completion.choices[0].message.content

    response = '[' + time_range + '] ' + topic_description + '\n' + response + '\n=================\n'
    return response

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    # Save the uploaded file
    file_path = f"tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    print(f"Received file: {file.filename}")

    reader = SimpleDirectoryReader(
        input_files=[file_path]
    )
    docs = reader.load_data()
    text = " ".join(doc.text for doc in docs)

    query = topics_prompt.format(transcript=text)

    messages = [
      {"role": "user", "content": query},
    ]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
      temperature=0,
      messages=messages,
      response_format={"type": "json_object"}
    )
    topics = json.loads(completion.choices[0].message.content)
    print(f"Produced list of topics:\n\n{topics}")
    
    summaries = []
    tasks = []

    for topic in topics['topics']:
        time_range = topic['time_range']
        topic_description = topic['topic_description']
        tasks.append(get_summary(client, text, time_range, topic_description, summary_prompt))

    summaries = await asyncio.gather(*tasks)
    print("Received topic summaries now saving to file.")
    # Join all summaries into a single string
    summaries_str = "\n".join(summaries)

    # Get the stem of the original file name
    file_stem = Path(file.filename).stem

    # Define the path to the summary notes file
    output_file_path = f"./tmp/podcast_summary_notes.txt"

    # Write the summaries to the file
    with open(output_file_path, "w") as output_file:
        output_file.write(summaries_str)

    # Remove the scheduling of temporary files for deletion
    # Instead, delete the files immediately after sending the response
    try:
        os.remove(file_path)
        print(f"Successfully removed temporary file: {file_path}")
    except Exception as e:
        print(f"Error removing temporary file: {file_path}. Exception: {e}")

    # Return the file using FileResponse
    return FileResponse(path=output_file_path, filename=f"podcast_summary_notes.txt")