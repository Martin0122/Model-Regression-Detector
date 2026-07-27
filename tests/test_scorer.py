from src.scoring import scorer
from src.config import EvalRun, RunMetadata
from src.services.load_configs import load_prompt_config
from datetime import datetime, timezone
import asyncio
from pathlib import Path

def save_run(eval_run: EvalRun, output_dir: str = "./src/runs") -> None:
    safe_id = eval_run.run_metadata.run_id.replace(':', '-')
    output_path = Path(output_dir) / f"{safe_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(eval_run.model_dump_json(indent=2))

async def main():
    config = load_prompt_config('./prompts/v1_classifier.yaml')
    results = await scorer()
    print("Successully scored evaluations!")
    print('=======================================================')

    eval_run = EvalRun(
        run_metadata=RunMetadata(
            run_id=f"run_{datetime.now(timezone.utc).isoformat()}",
            prompt_version=config.version_id,
            model=config.model,
            judge_model=config.model, # using the same as classifer for now
            timestamp=datetime.now(timezone.utc).isoformat()
        ),
        results=results
    )

    save_run(eval_run)

if __name__ == "__main__":
    asyncio.run(main())