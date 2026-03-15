import datetime


def metric_parser(stats, name):
    feedback = stats.get('feedback_stats', {}).get(name, {})


    # Format the metrics
    avg_score = feedback.get('avg', 0)
    stdev = feedback.get('stdev', 0)

    output = f"""
* **Feedback: {name}**
        - **Average Score:** {avg_score * 100:.1f}% ({avg_score:.3f})
        - **Standard Deviation:** {stdev:.3f}
        - **Evaluated Runs:** {feedback.get('n', 0)} (Errors: {feedback.get('errors', 0)})
    """
    return output.strip()


def beautify_langsmith_stats(stats: dict) -> str:
    """
    Parses a LangSmith stats dictionary and formats it into clean Markdown.
    Safely handles datetime and timedelta objects.
    """

    output = """### Evaluation Summary"""
    for key in stats.get('feedback_stats', {}).keys():
        output = "\n".join([output, metric_parser(stats, key)])

    runs = stats.get('run_stats', {})
    p99 = runs.get('latency_p99')
    p99_str = f"{p99.total_seconds():.2f}s" if isinstance(p99, datetime.timedelta) else "N/A"
    p50 = runs.get('latency_p50')
    p50_str = f"{p50.total_seconds():.2f}s" if isinstance(p50, datetime.timedelta) else "N/A"

    last_run = runs.get('last_run_start_time')
    last_run_str = last_run.strftime("%B %d, %Y at %I:%M %p") if isinstance(last_run, datetime.datetime) else "N/A"

    # Build the Markdown output
    run_output = \
        f"""
* **Run Performance & Telemetry**
        - **Total Runs:** {runs.get('run_count', 0)}
        - **Latency:** {p50_str} (Median) | {p99_str} (99th Percentile)
        - **Token Usage:** {runs.get('total_tokens', 0):,} Total ({runs.get('prompt_tokens', 0):,} prompt / {runs.get('completion_tokens', 0):,} completion)
        - **Last Run:** {last_run_str}
        """

    output = "\n".join([output, run_output])
    return output.strip()