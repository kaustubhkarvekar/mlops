import json, wandb
from train import trainer, test_dataset, id2label
from sklearn.metrics import classification_report

eval_results = trainer.evaluate()
print(eval_results)

wandb.log({
    "final/loss":     eval_results["eval_loss"],
    "final/accuracy": eval_results["eval_accuracy"],
    "final/f1":       eval_results["eval_f1"],
})

preds  = trainer.predict(test_dataset).predictions.argmax(-1)
labels = [item["labels"].item() for item in test_dataset]

report = classification_report(labels, preds, target_names=list(id2label.values()), output_dict=True)

with open("eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

artifact = wandb.Artifact("eval-report", type="evaluation")
artifact.add_file("eval_report.json")
wandb.log_artifact(artifact)
wandb.finish()

