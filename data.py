import json

root = 'data'

evaluation_challenges_path = root + '/arc-agi_evaluation_challenges.json'
evaluation_solutions_path = root + '/arc-agi_evaluation_solutions.json'
test_challenges_path = root + '/arc-agi_test_challenges.json'
train_challenges_path = root + '/arc-agi_training_challenges.json'
train_solutions_path = root + '/arc-agi_training_solutions.json'
sample_submission_path = root + '/sample_submission.json'

def load_json(file_path):
    """Loads a JSON file from the given path."""
    with open(file_path, 'r') as f:
        return json.load(f)