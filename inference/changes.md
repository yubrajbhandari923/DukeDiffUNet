Make following changes to the train code to turn into inference.

Update the resume:  in config
Update the val_jsonl in config

remove train_engine.run() and only run val_evaluator.run()
save images, labels and predictions, make sure that predictions are not one-hot encoded.

turn on inference_mode in config

Remove Train tag and add Inference tag
Fix the save_dir