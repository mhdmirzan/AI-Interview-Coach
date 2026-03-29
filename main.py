import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from agents.coach import InterviewCoach

console = Console()

def run_cli():
    parser = argparse.ArgumentParser(description="AI Interview Coach")
    parser.add_argument("--jobs-dir", default="data/job_descriptions", help="Directory with job description files")
    parser.add_argument("--job", "-j", help="Backward-compatible single job file path")
    parser.add_argument("--type", "-t", default="technical", help="Interview type")
    parser.add_argument("--level", "-l", default="senior", help="Position level")
    parser.add_argument("--questions", "-q", type=int, default=5, help="Number of questions")
    args = parser.parse_args()

    jobs_dir = args.jobs_dir
    if args.job:
        jobs_dir = str(Path(args.job).parent)

    console.print(Panel.fit(
        "[bold cyan]AI Interview Coach[/bold cyan]\n"
        "Practice technical interviews with AI feedback",
        border_style="cyan"
    ))

    # Initialize coach
    coach = InterviewCoach(
        job_descriptions_dir=jobs_dir,
        interview_type=args.type,
        level=args.level,
        max_questions=args.questions,
    )

    session_id = "cli_session"
    topics = ["Python", "system design", "algorithms", "best practices", "behavioral"]

    # Start interview
    welcome = coach.start_interview(session_id, topics[:args.questions])
    console.print(f"\n[bold green]Interviewer:[/bold green] {welcome}\n")

    while True:
        try:
            answer = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Interview interrupted. Exiting...[/yellow]")
            break

        if not answer or not answer.strip():
            continue

        if answer.lower() in ['quit', 'exit', 'q']:
            console.print("[yellow]Interview ended early.[/yellow]")
            break

        result = coach.submit_answer(session_id, answer)

        # Show feedback
        feedback = result["feedback"]
        console.print(f"\n[dim]Score: {feedback.score}/10 - {feedback.understanding}[/dim]")

        if result["is_complete"]:
            # Generate and display report
            console.print("\n[bold]Generating your interview report...[/bold]\n")

            report = coach.generate_report(session_id)

            console.print(Panel(
                f"[bold]Overall Score: {report.overall_score}/10[/bold]\n"
                f"Recommendation: [cyan]{report.recommendation.upper()}[/cyan]\n\n"
                f"{report.summary}\n\n"
                f"[green]Strengths:[/green]\n" +
                "\n".join(f"  • {s}" for s in report.strengths) + "\n\n"
                f"[yellow]Areas to Improve:[/yellow]\n" +
                "\n".join(f"  • {a}" for a in report.areas_to_improve),
                title="Interview Report",
                border_style="green"
            ))
            break

        console.print(f"\n[bold green]Interviewer:[/bold green] {result['next_question']}\n")
        console.print(f"[dim]({result['questions_remaining']} questions remaining)[/dim]\n")

if __name__ == "__main__":
    run_cli()