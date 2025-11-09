from data import *
from utils import get_small_sample
from models import ProgramSynthesisModel, DummyModel
from evaluate import make_predictions

def main():

    print("Loading data...")
    evaluation_challenges = load_json(evaluation_challenges_path)
    evaluation_solutions = load_json(evaluation_solutions_path)
    test_challenges = load_json(test_challenges_path)
    train_challenges = load_json(train_challenges_path)
    train_solutions = load_json(train_solutions_path)
    sample_submission = load_json(sample_submission_path)

    small_evaluation_challenges, small_evaluation_solutions = get_small_sample(evaluation_challenges, evaluation_solutions)


    model_bfs = ProgramSynthesisModel()
    _ = make_predictions(
        small_evaluation_challenges, model_bfs, small_evaluation_solutions, verbose=True
    )

if __name__ == "__main__":
    main()