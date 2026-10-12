"""Cloud metadata only: no model downloads, private media or API key output."""
import json
import os
from pathlib import Path

if os.environ.get("GITHUB_ACTIONS") != "true":
    raise SystemExit("Run on CI only")
import qai_hub as hub

client = hub.Client(hub.ClientConfig(api_token=os.environ["QAI_HUB_API_TOKEN"]))
found = 0
for summary in client.get_job_summaries(limit=20):
    if not summary.name.startswith("Recall_D3Nano_"):
        continue
    found += 1
    print(json.dumps({"id": summary.job_id, "name": summary.name,
                      "status": str(summary.status), "url": summary.url}), flush=True)
    if summary.status.failure:
        job = client.get_job(summary.job_id)
        logs = job.download_job_logs("/tmp/recall-qnn-inspect/" + summary.job_id)
        for path in logs:
            p = Path(path)
            if p.suffix in (".log", ".txt"):
                print(p.name + ":\n" + p.read_text(errors="replace")[-12000:])
print("Matching recent conversion jobs:", found)
