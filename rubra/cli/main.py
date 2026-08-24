import typer
from rich.console import Console
from rubra.__version__ import __version__

app = typer.Typer(
    name="rubra",
    help="Rubra — Agentic evaluation framework. Every aspect, nothing missed.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version():
    """Show Rubra version."""
    console.print(f"[bold red]Rubra[/bold red] v{__version__}")


@app.command()
def eval(
    trace_id: str = typer.Argument(None, help="Trace ID to evaluate"),
    metrics: str = typer.Option("all", help="Metric set: all, execution, tool, safety, quality, goal"),
):
    """Evaluate a captured trace."""
    from rubra.core.storage.db import auto_init_storage
    from rubra.core.evaluator.evaluator import evaluate as run_eval

    storage = auto_init_storage()

    if trace_id:
        trace = storage.get_trace(trace_id)
        if not trace:
            console.print(f"[red]Trace {trace_id} not found.[/red]")
            raise typer.Exit(1)
        traces = [trace]
    else:
        traces = storage.list_traces(limit=1)
        if not traces:
            console.print("[yellow]No traces found. Run an instrumented agent first.[/yellow]")
            raise typer.Exit(0)

    for t in traces:
        report = run_eval(t, metrics=metrics, persist=True)
        console.print()
        console.print(f"[bold red]Rubra Eval Report[/bold red]")
        console.print(f"Agent: [bold]{report.agent_name}[/bold]  |  Trace: {report.trace_id[:8]}…")
        console.print(f"Task: {report.task or 'N/A'}")
        console.print()

        from rich.table import Table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="dim", width=40)
        table.add_column("Score", justify="right", width=10)
        table.add_column("Result", width=8)
        table.add_column("Reason", width=50)

        for r in report.results:
            score_str = f"{r.score:.4f}" if r.score is not None else "N/A"
            result_str = "[green]PASS[/green]" if r.passed else ("[red]FAIL[/red]" if r.passed is False else "[dim]N/A[/dim]")
            reason_short = (r.reason or "")[:50]
            table.add_row(r.metric_name, score_str, result_str, reason_short)

        console.print(table)
        console.print()

        if report.rubra_score is not None:
            console.print(f"[bold]Rubra Score:[/bold] {report.rubra_score:.4f}")
        if report.tool_intelligence_score is not None:
            console.print(f"[bold]Tool Intelligence:[/bold] {report.tool_intelligence_score:.4f}")
        if report.agentic_efficiency_score is not None:
            console.print(f"[bold]Agentic Efficiency:[/bold] {report.agentic_efficiency_score:.4f}")


@app.command()
def traces(
    limit: int = typer.Option(20, help="Number of traces to list"),
    agent: str = typer.Option(None, help="Filter by agent name"),
):
    """List recent traces."""
    from rubra.core.storage.db import auto_init_storage

    storage = auto_init_storage()
    trace_list = storage.list_traces(agent_name=agent, limit=limit)

    if not trace_list:
        console.print("[yellow]No traces found.[/yellow]")
        return

    from rich.table import Table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Trace ID", width=10)
    table.add_column("Agent", width=20)
    table.add_column("Status", width=12)
    table.add_column("Tools", width=6, justify="right")
    table.add_column("Duration ms", width=12, justify="right")
    table.add_column("Task", width=40)

    for t in trace_list:
        status_style = "green" if t.status.value == "completed" else "red"
        table.add_row(
            t.trace_id[:8] + "…",
            t.agent_name,
            f"[{status_style}]{t.status.value}[/{status_style}]",
            str(t.total_tool_calls),
            f"{t.duration_ms:.0f}" if t.duration_ms else "N/A",
            (t.task or "")[:40],
        )

    console.print(table)


@app.command()
def report(
    trace_id: str = typer.Argument(None, help="Trace ID to report on"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    metrics: str = typer.Option("all", help="Metric set: all, execution, tool, safety, quality, goal"),
):
    """Generate a self-contained HTML evaluation report."""
    from rubra.core.storage.db import auto_init_storage
    from rubra.core.evaluator.evaluator import evaluate as run_eval

    storage = auto_init_storage()

    if trace_id:
        trace = storage.get_trace(trace_id)
        if not trace:
            console.print(f"[red]Trace {trace_id} not found.[/red]")
            raise typer.Exit(1)
    else:
        traces = storage.list_traces(limit=1)
        if not traces:
            console.print("[yellow]No traces found. Run an instrumented agent first.[/yellow]")
            raise typer.Exit(0)
        trace = traces[0]

    eval_report = run_eval(trace, metrics=metrics, persist=False)

    if output is None:
        output = f"rubra_report_{trace.trace_id[:8]}.html"

    eval_report.to_html(path=output)
    console.print(f"[bold green]Report saved:[/bold green] {output}")
    console.print(f"  Agent:   {eval_report.agent_name}")
    console.print(f"  Metrics: {eval_report.total_metrics}  Pass: {eval_report.passed}  Fail: {eval_report.failed}")
    if eval_report.rubra_score is not None:
        console.print(f"  [bold]Rubra Score: {eval_report.rubra_score:.4f}[/bold]")


if __name__ == "__main__":
    app()
