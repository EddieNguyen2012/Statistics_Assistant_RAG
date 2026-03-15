import dash
from dash import html, dcc, Input, Output, State
import time
from src.pipeline import pipeline_enforced
from langchain_ollama import ChatOllama



# Initialize the Dash app
app = dash.Dash(__name__, title="RAG Assistant")
model = ChatOllama(model="gemma3", format="json", temperature=0)
chain = pipeline_enforced(model)


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
    time.sleep(2)

    try:
        response = chain.invoke(user_query)
        # response is now a RAGResponse (Pydantic object)
        generated_answer = response.to_str()
        retrieved_sources = response.citations
        retrieved_sources = [row.to_str() for row in retrieved_sources]
    except Exception as e:
        generated_answer = str(e)
        retrieved_sources = ["None"]

    return generated_answer, retrieved_sources

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'}, children=[

        html.H1("Statistical Methods Assistant (RAG)", style={'textAlign': 'center'}),
        html.Hr(),

        # Input Section
        html.Div([
            html.Label("Ask a question:", style={'fontWeight': 'bold'}),
            dcc.Textarea(
                id='user-input',
                className='user-input',
                placeholder='Type your question here...',
                style={'width': '100%', 'height': '100px', 'marginTop': '10px', 'padding': '10px'}
            ),
            html.Button(
                'Submit Query',
                id='submit-button',
                className='submit-btn',
                n_clicks=0,

            ),
        ]),

        html.Br(),

        dcc.Loading(
            id="loading-spinner",
            type="default",
            children=[
                html.Div(id='rag-output-container', children=[

                    # Answer Display
                    html.Div(
                        [
                        html.H3("Answer:"),
                        html.Div(children=[dcc.Markdown(id='answer-output', className='answer-content')],
                                 className='answer-box')

                    ]),
                    html.Div([
                        html.H4("Retrieved Sources:"),
                        html.Ul(id='sources-output', style={'color': '#555'})
                                ], style={'marginTop': '20px'})

                ], style={'display': 'none'})  # Hidden until first query
            ]
        )
    ])


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
    This callback function processes a user query and returns a response from a RAG system.
    It updates the UI with the styling of a container, the answer from the system, and a list of sources.
    """
    if not user_query or user_query.strip() == "":
        return {'display': 'block'}, "Please enter a valid question.", []

    answer, sources = query_your_rag_system(user_query)

    safe_answer = str(answer)

    if sources and sources != ["None"]:
        sources_html = [html.Li(source) for source in sources]
    else:
        sources_html = [html.Li("No sources found.")]

    # Return: 1. Show container, 2. The safe string answer, 3. The HTML list items
    return {'display': 'block'}, safe_answer, sources_html


# --- 4. RUN SERVER ---
if __name__ == '__main__':
    app.run(debug=False)