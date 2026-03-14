import dash
from dash import html, dcc, Input, Output, State
import time
from src.pipeline import pipeline
import json
from langchain_community.chat_models import ChatOllama
# from langchain_community.chat_models import ChatOpenAI
# import getpass
# import os


# Initialize the Dash app
app = dash.Dash(__name__, title="RAG Assistant")

json_format = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "answer": {
      "type": "string",
      "description": "Final answer for the user. MUST include in-text citations like [S1], [S2]. LaTeX is allowed using \\( ... \\) or \\[ ... \\]."
    },
    "citations_used": {
      "type": "array",
      "description": "Unique list of source IDs that were cited in the answer (order of first appearance).",
      "items": {
        "type": "string",
        "pattern": "^S\\d+$"
      }
    },
    "citation_spans": {
      "type": "array",
      "description": "Optional: machine-readable citation locations for highlighting. Offsets refer to character indices in `answer`.",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "source_id": {
            "type": "string",
            "pattern": "^S\\d+$"
          },
          "start": { "type": "integer", "minimum": 0 },
          "end": { "type": "integer", "minimum": 0 },
          "note": {
            "type": "string",
            "description": "Optional note about what the span supports."
          }
        },
        "required": ["source_id", "start", "end"]
      }
    },
    "confidence": {
      "type": "string",
      "enum": ["low", "medium", "high"],
      "description": "Overall confidence given the provided sources."
    }
  },
  "required": ["answer", "citations_used", "confidence"]
}
# --- 1. DUMMY RAG FUNCTION ---
# Replace the contents of this function with your actual RAG logic
# (e.g., LangChain, LlamaIndex, OpenAI API calls, Vector DB queries).
def query_your_rag_system(user_query):
    """
    Handles a user query by passing it through the RAG pipeline.
    Simulates processing time and invokes the pipeline with a ChatOllama model.

    Args:
        user_query (str): The question asked by the user.

    Returns:
        tuple: (generated_answer, retrieved_sources)
            generated_answer (str): The response from the LLM.
            retrieved_sources (list[str]): A list of strings representing the sources used.
    """
    # Simulating network/LLM processing time
    time.sleep(2)
    model = ChatOllama(model="deepseek-r1", format="json", temperature=0)
    # model = ChatOllama(model="gpt-oss", format="json", temperature=0)
    # model = ChatOpenAI(model="gpt-5-nano", temperature=0)
    # Dummy outputs

    try:
        generated_answer = pipeline(model).invoke(user_query)

        retrieved_sources = [
            "Source 1: DocumentA.pdf (Page 3)",
            "Source 2: internal_wiki_article_42.html",
            "Source 3: database_record_id_992"
        ]
    except Exception as e:
        generated_answer = e
        retrieved_sources = ["None"]

    return generated_answer, retrieved_sources


# --- 2. APP LAYOUT ---
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'}, children=[

        html.H1("Knowledge Base Assistant (RAG)", style={'textAlign': 'center'}),
        html.Hr(),

        # Input Section
        html.Div([
            html.Label("Ask a question:", style={'fontWeight': 'bold'}),
            dcc.Textarea(
                id='user-input',
                placeholder='Type your question here...',
                style={'width': '100%', 'height': '100px', 'marginTop': '10px', 'padding': '10px'}
            ),
            html.Button(
                'Submit Query',
                id='submit-button',
                n_clicks=0,
                style={'marginTop': '10px', 'padding': '10px 20px', 'cursor': 'pointer', 'backgroundColor': '#007BFF',
                       'color': 'white', 'border': 'none', 'borderRadius': '5px'}
            ),
        ]),

        html.Br(),

        # Output Section wrapped in a Loading component
        dcc.Loading(
            id="loading-spinner",
            type="default",
            children=[
                html.Div(id='rag-output-container', children=[

                    # Answer Display
                    html.Div([
                        html.H3("Answer:"),
                        html.Div(id='answer-output',
                                 style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px',
                                        'minHeight': '50px'})
                    ]),

                    # Sources Display
                    html.Div([
                        html.H4("Retrieved Sources:"),
                        html.Ul(id='sources-output', style={'color': '#555'})
                    ], style={'marginTop': '20px'})

                ], style={'display': 'none'})  # Hidden until first query
            ]
        )
    ])


# --- 3. CALLBACK ---
@app.callback(
    [Output('rag-output-container', 'style'),
     Output('answer-output', 'children'),
     Output('sources-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value')],
    prevent_initial_call=True
)
def process_query(n_clicks, user_query):
    """
    Dash callback to process the user's question when the submit button is clicked.
    Updates the UI with the generated answer and retrieved sources.

    Args:
        n_clicks (int): Number of times the submit button has been clicked.
        user_query (str): The text input from the user.

    Returns:
        tuple: (display_style, answer_content, sources_list)
            display_style (dict): CSS style to show the output container.
            answer_content (str): The answer to be displayed.
            sources_list (list): A list of html.Li components for the sources.
    """
    if not user_query or user_query.strip() == "":
        return {'display': 'block'}, "Please enter a valid question.", []

    # Call your RAG function here
    answer, sources = query_your_rag_system(user_query)

    # Format the sources into a list of HTML list items
    sources_html = [html.Li(source) for source in sources]

    # Return: 1. Show container, 2. The Answer, 3. The Sources
    return {'display': 'block'}, answer, sources_html


# --- 4. RUN SERVER ---
if __name__ == '__main__':
    app.run(debug=True)